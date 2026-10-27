#include <algorithm>
#include <cstring>
#include <limits>
#include <map>
#include <optional>
#include <vector>

#include "core/quant.hpp"
#include "npu/gemv.hpp"
#include "npu/xrt_kernel.hpp"

namespace gufo_npu::npu {

namespace {

const char* PrologueName(Prologue p) {
  switch (p) {
    case Prologue::kPlain:
      return "plain";
    case Prologue::kAct:
      return "act";
    case Prologue::kResNorm:
      return "resnorm";
  }
  return "";
}

const char* EpilogueName(Epilogue e) {
  return e == Epilogue::kSwiGlu ? "swiglu" : "store";
}

}  // namespace

std::string Q4nxGemvSequence(std::size_t dim, std::size_t padded_rows,
                             Prologue prologue, Epilogue epilogue) {
  return "q4nx_gemv_k" + std::to_string(dim) + "_n" +
         std::to_string(padded_rows) + "_" + PrologueName(prologue) + "_" +
         EpilogueName(epilogue);
}

struct GemvRuntime::Impl {
  std::filesystem::path dir;
  HwContext hw;
  std::map<std::string, std::unique_ptr<Sequence>> sequences;

  explicit Impl(const std::filesystem::path& d) : dir(d), hw(d, "q4nx_gemv") {}

  Sequence& Get(const std::string& name) {
    auto& slot = sequences[name];
    if (!slot) {
      if (!std::filesystem::exists(dir / (name + ".insts.elf"))) {
        throw std::runtime_error("no q4nx GEMV sequence " + name + " in " +
                                 dir.string());
      }
      slot = std::make_unique<Sequence>(hw, dir, name);
    }
    return *slot;
  }
};

GemvRuntime::GemvRuntime(const std::filesystem::path& kernel_dir)
    : impl_(std::make_unique<Impl>(kernel_dir)) {}

GemvRuntime::~GemvRuntime() = default;

struct Buffer::Impl {
  xrt::bo bo;
  std::size_t bytes;
};

Buffer::Buffer(GemvRuntime& runtime, std::size_t bytes)
    : impl_(std::make_shared<Impl>(Impl{
          runtime.impl().hw.Buffer(std::max<std::size_t>(bytes, 64)), bytes})) {
  std::memset(data(), 0, bytes);
}

Buffer::Buffer(std::shared_ptr<Impl> impl) : impl_(std::move(impl)) {}
Buffer::~Buffer() = default;
Buffer::Buffer(const Buffer&) = default;
Buffer& Buffer::operator=(const Buffer&) = default;

Buffer Buffer::Sub(std::size_t offset, std::size_t bytes) const {
  if (offset % 64 != 0 || offset + bytes > impl_->bytes) {
    throw std::invalid_argument("npu buffer: sub-buffer out of range");
  }
  return Buffer(
      std::make_shared<Impl>(Impl{xrt::bo(impl_->bo, bytes, offset), bytes}));
}

void* Buffer::data() {
  return impl_->bo.map<void*>();
}

std::size_t Buffer::size() const {
  return impl_->bytes;
}

void Buffer::ToDevice() {
  impl_->bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
}

void Buffer::FromDevice() {
  impl_->bo.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
}

struct Q4nxGemv::Impl {
  Sequence& sequence;
  Prologue prologue;
  std::size_t w_bytes;
  xrt::bo w;
  // Buffers of the host-memory convenience Run.
  std::optional<Buffer> x;
  std::optional<Buffer> y;
  std::optional<Buffer> residual;
  std::optional<xrt::run> run;

  Impl(GemvRuntime::Impl& rt, const q4nx::Matrix& m, Prologue p, Epilogue e)
      : sequence(rt.Get(Q4nxGemvSequence(m.dim, m.padded_rows, p, e))),
        prologue(p),
        w_bytes(m.bytes.size()),
        w(rt.hw.Buffer(w_bytes)) {
    std::memcpy(w.map<void*>(), m.bytes.data(), w_bytes);
    w.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  }

  xrt::run Bind(Buffer& x_buf, Buffer& y_buf, Buffer& r_buf) {
    return sequence.Run(
        {&x_buf.impl().bo, &w, &y_buf.impl().bo, &r_buf.impl().bo});
  }
};

Q4nxGemv::Q4nxGemv(GemvRuntime& runtime, const q4nx::Matrix& weights,
                   Prologue prologue, Epilogue epilogue)
    : impl_(
          std::make_unique<Impl>(runtime.impl(), weights, prologue, epilogue)),
      dim_(weights.dim),
      rows_(weights.rows),
      padded_rows_(weights.padded_rows) {
  if (weights.dim > kGemvMaxDim) {
    throw std::invalid_argument("q4nx GEMV: dim above the design's maximum");
  }
  impl_->x.emplace(runtime, kGemvInputBytes);
  impl_->y.emplace(runtime, padded_rows_ * sizeof(float));
  impl_->residual.emplace(runtime, kGemvHidden * sizeof(float));
}

Q4nxGemv::~Q4nxGemv() = default;

void Q4nxGemv::Run(Buffer& x, Buffer& y, Buffer& residual) {
  impl_->run.emplace(impl_->Bind(x, y, residual));
  impl_->run->start();
  impl_->run->wait();
}

void Q4nxGemv::Run(const float* x, float* y) {
  std::memcpy(impl_->x->data(), x, dim_ * sizeof(float));
  impl_->x->ToDevice();
  if (!impl_->run) {
    impl_->run.emplace(impl_->Bind(*impl_->x, *impl_->y, *impl_->residual));
  }
  impl_->run->start();
  impl_->run->wait();
  impl_->y->FromDevice();
  std::memcpy(y, impl_->y->data(), rows_ * sizeof(float));
}

void Q4nxGemv::Run(const std::uint16_t* x, float* y) {
  std::vector<float> wide(dim_);
  for (std::size_t i = 0; i < dim_; ++i) {
    wide[i] = quant::Bf16ToFloat(x[i]);
  }
  std::memcpy(impl_->x->data(), wide.data(), dim_ * sizeof(float));
  impl_->x->ToDevice();
  if (!impl_->run) {
    impl_->run.emplace(impl_->Bind(*impl_->x, *impl_->y, *impl_->residual));
  }
  impl_->run->start();
  impl_->run->wait();
  impl_->y->FromDevice();
  std::memcpy(y, impl_->y->data(), padded_rows_ * sizeof(float));
}

Q4nxGemv::Timing Q4nxGemv::Time(int iterations) {
  iterations = std::max(iterations, 1);
  if (!impl_->run) {
    impl_->run.emplace(impl_->Bind(*impl_->x, *impl_->y, *impl_->residual));
  }
  Timing t;
  t.best_us = std::numeric_limits<double>::max();
  double total = 0;
  for (int i = 0; i < iterations; ++i) {
    const auto start = Clock::now();
    impl_->run->start();
    impl_->run->wait();
    const double us = MicrosSince(start);
    total += us;
    t.best_us = std::min(t.best_us, us);
  }
  t.mean_us = total / iterations;
  return t;
}

std::size_t Q4nxGemv::weight_bytes() const {
  return impl_->w_bytes;
}

struct Chain::Impl {
  xrt::runlist runlist;
  std::vector<xrt::run> runs;
};

Chain::Chain(GemvRuntime& runtime)
    : impl_(std::make_unique<Impl>(
          Impl{xrt::runlist(*runtime.impl().hw.context), {}})) {}

Chain::~Chain() = default;

void Chain::Add(Q4nxGemv& gemv, Buffer& x, Buffer& y, Buffer& residual) {
  impl_->runs.push_back(gemv.impl().Bind(x, y, residual));
  impl_->runlist.add(impl_->runs.back());
}

void Chain::Run() {
  impl_->runlist.execute();
  impl_->runlist.wait();
}

}  // namespace gufo_npu::npu
