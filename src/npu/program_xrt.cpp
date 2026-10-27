#include <algorithm>
#include <cstring>
#include <map>
#include <stdexcept>
#include <vector>

#include "npu/buffer_xrt.hpp"
#include "npu/program.hpp"
#include "npu/xrt_kernel.hpp"

namespace gufo_npu::npu {

struct Program::Impl {
  std::filesystem::path dir;
  std::string name;
  HwContext hw;
  // Calls pass their sequence's instruction words in a host buffer (the
  // instruction-buffer flow), shared by every call of the sequence; the
  // firmware adds the argument addresses at run time.
  xrt::kernel kernel;
  std::map<std::string, std::unique_ptr<InstrSequence>> sequences;

  Impl(const std::filesystem::path& d, const std::string& n)
      : dir(d), name(n), hw(d, n), kernel(*hw.context, hw.kernel_name) {}

  InstrSequence& Get(const std::string& sequence) {
    auto& slot = sequences[sequence];
    if (!slot) {
      const auto path = dir / (sequence + ".insts.bin");
      if (!std::filesystem::exists(path)) {
        sequences.erase(sequence);
        throw std::runtime_error("no sequence " + sequence + " for " + name +
                                 " in " + dir.string());
      }
      slot = std::make_unique<InstrSequence>(hw, kernel, path);
    }
    return *slot;
  }
};

Program::Program(const std::filesystem::path& dir, const std::string& name)
    : impl_(std::make_unique<Impl>(dir, name)) {}

Program::~Program() = default;

const std::string& Program::name() const {
  return impl_->name;
}

bool Program::Has(const std::string& sequence) const {
  return std::filesystem::exists(impl_->dir / (sequence + ".insts.bin"));
}

Buffer::Buffer(Program& program, std::size_t bytes)
    : impl_(std::make_shared<Impl>(Impl{
          program.impl().hw.Buffer(std::max<std::size_t>(bytes, 64)), bytes})) {
  // As the GEMV runtime's buffers: zeroed, and the zeros pushed out of the
  // CPU cache before the NPU writes it.
  std::memset(data(), 0, bytes);
  ToDevice();
}

struct Batch::Impl {
  // Calls per runlist: a chained submission lives in a 4 KB command buffer
  // (the driver's MAX_CHAIN_CMDBUF_SIZE), which holds a few dozen.
  static constexpr std::size_t kChain = 24;

  Program::Impl& program;
  std::vector<xrt::run> runs;
  std::vector<Buffer> keep;  // the runs' buffers, alive while they are
  std::vector<xrt::runlist> lists;
  bool sealed = false;

  explicit Impl(Program::Impl& p) : program(p) {}

  void Seal() {
    lists.clear();
    for (std::size_t i = 0; i < runs.size(); i += kChain) {
      lists.emplace_back(*program.hw.context);
      for (std::size_t j = i; j < std::min(runs.size(), i + kChain); ++j) {
        lists.back().add(runs[j]);
      }
    }
    sealed = true;
  }
};

Batch::Batch(Program& program)
    : impl_(std::make_unique<Impl>(program.impl())) {}

Batch::~Batch() = default;

void Batch::Add(const std::string& sequence,
                std::initializer_list<const Buffer*> args) {
  InstrSequence& seq = impl_->program.Get(sequence);
  xrt::run run(impl_->program.kernel);
  run.set_arg(0, kOpcodeRun);
  run.set_arg(1, seq.bo);
  run.set_arg(2, static_cast<std::uint32_t>(seq.words.size()));
  int index = 3;
  for (const Buffer* arg : args) {
    run.set_arg(index++, arg->impl().bo);
    impl_->keep.push_back(*arg);
  }
  impl_->runs.push_back(std::move(run));
  impl_->sealed = false;
}

std::size_t Batch::size() const {
  return impl_->runs.size();
}

void Batch::Run() {
  if (impl_->runs.empty()) {
    return;
  }
  if (!impl_->sealed) {
    impl_->Seal();
  }
  // One runlist in flight at a time; the context runs them in order.
  for (auto& list : impl_->lists) {
    list.execute();
    list.wait();
  }
}

void Batch::Clear() {
  impl_->lists.clear();
  impl_->runs.clear();
  impl_->keep.clear();
  impl_->sealed = false;
}

}  // namespace gufo_npu::npu
