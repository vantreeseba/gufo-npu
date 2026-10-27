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
  std::map<std::string, std::unique_ptr<Sequence>> sequences;
  // Runs of finished batches, per sequence, for the next ones: making a run
  // allocates its command buffer on the device, ~10-30 µs, more than a
  // small call's own NPU time.
  std::map<const Sequence*, std::vector<xrt::run>> spare;

  Impl(const std::filesystem::path& d, const std::string& n)
      : dir(d), name(n), hw(d, n) {}

  Sequence& Get(const std::string& sequence) {
    auto& slot = sequences[sequence];
    if (!slot) {
      if (!std::filesystem::exists(dir / (sequence + ".insts.elf"))) {
        sequences.erase(sequence);
        throw std::runtime_error("no sequence " + sequence + " for " + name +
                                 " in " + dir.string());
      }
      slot = std::make_unique<Sequence>(hw, dir, sequence);
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
  return std::filesystem::exists(impl_->dir / (sequence + ".insts.elf"));
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
  std::vector<const Sequence*> sequences;  // of each run
  std::vector<Buffer> keep;  // the runs' buffers, alive while they are
  std::vector<xrt::runlist> lists;
  bool sealed = false;

  explicit Impl(Program::Impl& p) : program(p) {}
  ~Impl() { Release(); }

  // Drops the runlists and hands the runs back to the program.
  void Release() {
    lists.clear();
    for (std::size_t i = 0; i < runs.size(); ++i) {
      program.spare[sequences[i]].push_back(std::move(runs[i]));
    }
    runs.clear();
    sequences.clear();
    keep.clear();
    sealed = false;
  }

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
  Sequence& seq = impl_->program.Get(sequence);
  auto& spare = impl_->program.spare[&seq];
  xrt::run run = spare.empty() ? xrt::run(seq.kernel) : std::move(spare.back());
  if (!spare.empty()) {
    spare.pop_back();
  }
  run.set_arg(0, kOpcodeRun);
  run.set_arg(1, 0);
  run.set_arg(2, 0);
  int index = 3;
  for (const Buffer* arg : args) {
    run.set_arg(index++, arg->impl().bo);
    impl_->keep.push_back(*arg);
  }
  impl_->runs.push_back(std::move(run));
  impl_->sequences.push_back(&seq);
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
  impl_->Release();
}

}  // namespace gufo_npu::npu
