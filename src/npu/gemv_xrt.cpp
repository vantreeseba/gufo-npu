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
                             Prologue prologue, Epilogue epilogue,
                             std::size_t tokens, const std::string& design) {
  return design + "_k" + std::to_string(dim) + "_n" +
         std::to_string(padded_rows) + "_" + PrologueName(prologue) + "_" +
         EpilogueName(epilogue) + "_t" + std::to_string(tokens);
}

struct GemvRuntime::Impl {
  std::filesystem::path dir;
  Flow flow;
  GemvDesign design;
  HwContext hw;
  // The kernel of the instruction-buffer flow.
  std::optional<xrt::kernel> kernel;
  std::map<std::string, std::unique_ptr<Sequence>> sequences;
  // Instruction-buffer flow: one buffer per variant, shared by its calls.
  std::map<std::string, std::unique_ptr<InstrSequence>> instructions;

  Impl(const std::filesystem::path& d, Flow f, const GemvDesign& g)
      : dir(d), flow(f), design(g), hw(d, g.name) {
    if (flow == Flow::kInstructionBuffer) {
      kernel.emplace(*hw.context, hw.kernel_name);
    }
  }

  std::filesystem::path Path(const std::string& name, const char* suffix) {
    auto path = dir / (name + suffix);
    if (!std::filesystem::exists(path)) {
      throw std::runtime_error("no q4nx GEMV sequence " + name + " in " +
                               dir.string());
    }
    return path;
  }

  // Loads variant `name` for this runtime's flow; throws when it was not
  // compiled.
  void Require(const std::string& name) {
    if (flow == Flow::kElf) {
      auto& slot = sequences[name];
      if (!slot) {
        Path(name, ".insts.elf");
        slot = std::make_unique<Sequence>(hw, dir, name);
      }
    } else {
      auto& slot = instructions[name];
      if (!slot) {
        slot = std::make_unique<InstrSequence>(hw, *kernel,
                                               Path(name, ".insts.bin"));
      }
    }
  }

  // A run of variant `name` (Require()d) on `args`.
  xrt::run Run(const std::string& name,
               std::initializer_list<const xrt::bo*> args) {
    if (flow == Flow::kElf) {
      return sequences.at(name)->Run(args);
    }
    return instructions.at(name)->Run(*kernel, args);
  }
};

GemvRuntime::GemvRuntime(const std::filesystem::path& kernel_dir, Flow flow,
                         GemvDesign design)
    : design_(std::move(design)),
      impl_(std::make_unique<Impl>(kernel_dir, flow, design_)) {}

GemvRuntime::~GemvRuntime() = default;

Flow GemvRuntime::flow() const {
  return impl_->flow;
}

struct Buffer::Impl {
  xrt::bo bo;
  std::size_t bytes;
};

Buffer::Buffer(GemvRuntime& runtime, std::size_t bytes)
    : impl_(std::make_shared<Impl>(Impl{
          runtime.impl().hw.Buffer(std::max<std::size_t>(bytes, 64)), bytes})) {
  // Zero it and push the zeros out of the CPU cache now: dirty lines left
  // behind can be written back over what the NPU writes later (seen as
  // intermittently corrupted LM head output).
  std::memset(data(), 0, bytes);
  ToDevice();
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
  GemvRuntime::Impl& rt;
  std::string sequences[kGemvTokens];
  std::size_t w_bytes;
  xrt::bo w;
  // Buffers of the host-memory convenience Run.
  std::optional<Buffer> x;
  std::optional<Buffer> y;
  std::optional<Buffer> residual;
  std::optional<xrt::run> run;

  Impl(GemvRuntime::Impl& r, const q4nx::Matrix& m, Prologue p, Epilogue e)
      : rt(r), w_bytes(m.bytes.size()), w(rt.hw.Buffer(w_bytes)) {
    for (std::size_t t = 0; t < kGemvTokens; ++t) {
      sequences[t] =
          Q4nxGemvSequence(m.dim, m.padded_rows, p, e, t + 1, rt.design.name);
      rt.Require(sequences[t]);
    }
    std::memcpy(w.map<void*>(), m.bytes.data(), w_bytes);
    w.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  }

  xrt::run Bind(Buffer& x_buf, Buffer& y_buf, Buffer& r_buf,
                std::size_t tokens) {
    if (tokens < 1 || tokens > kGemvTokens) {
      throw std::invalid_argument("q4nx GEMV: 1 or 2 tokens per call");
    }
    return rt.Run(sequences[tokens - 1],
                  {&x_buf.impl().bo, &w, &y_buf.impl().bo, &r_buf.impl().bo});
  }
};

Q4nxGemv::Q4nxGemv(GemvRuntime& runtime, const q4nx::Matrix& weights,
                   Prologue prologue, Epilogue epilogue)
    : impl_(
          std::make_unique<Impl>(runtime.impl(), weights, prologue, epilogue)),
      dim_(weights.dim),
      rows_(weights.rows),
      padded_rows_(weights.padded_rows),
      epilogue_(epilogue) {
  const GemvDesign& design = runtime.design();
  if (weights.dim > design.max_dim) {
    throw std::invalid_argument("q4nx GEMV: dim above the design's maximum");
  }
  // Must match the design's variants (kernels/designs/q4nx_gemv.py, or a
  // model's own copy).
  if (epilogue == Epilogue::kSwiGlu) {
    // Compact bf16 tokens MAX_DIM apart, where the act input stage reads them.
    y_token_stride_ = design.max_dim / 2;
  } else if (padded_rows_ == design.hidden || rows_ == design.hidden) {
    y_token_stride_ = design.slot();
  } else {
    y_token_stride_ = padded_rows_;
  }
  impl_->x.emplace(runtime, design.input_bytes());
  impl_->y.emplace(runtime, y_bytes());
  impl_->residual.emplace(runtime, design.residual_bytes());
}

Q4nxGemv::~Q4nxGemv() = default;

std::size_t Q4nxGemv::y_bytes() const {
  if (epilogue_ == Epilogue::kSwiGlu) {
    return 8 * y_token_stride_ * sizeof(float);
  }
  return ((kGemvTokens - 1) * y_token_stride_ + padded_rows_) * sizeof(float);
}

void Q4nxGemv::Run(Buffer& x, Buffer& y, Buffer& residual, std::size_t tokens) {
  impl_->run.emplace(impl_->Bind(x, y, residual, tokens));
  impl_->run->start();
  impl_->run->wait();
}

void Q4nxGemv::Run(const float* x, float* y) {
  std::memcpy(impl_->x->data(), x, dim_ * sizeof(float));
  impl_->x->ToDevice();
  if (!impl_->run) {
    impl_->run.emplace(impl_->Bind(*impl_->x, *impl_->y, *impl_->residual, 1));
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
  Run(wide.data(), y);
}

Q4nxGemv::Timing Q4nxGemv::Time(int iterations) {
  iterations = std::max(iterations, 1);
  if (!impl_->run) {
    impl_->run.emplace(impl_->Bind(*impl_->x, *impl_->y, *impl_->residual, 1));
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

namespace {

// Word positions of the DDR address patches (opcode 0x81) of buffer argument
// `arg` in an instruction stream, in stream order (aie/Runtime/TxnEncoding.h).
std::vector<std::size_t> PatchSites(const std::vector<std::uint32_t>& w,
                                    std::uint32_t arg) {
  std::vector<std::size_t> sites;
  std::size_t pos = 4;  // after the header
  while (pos < w.size()) {
    const std::uint32_t op = w[pos] & 0xFF;
    std::size_t len = 0;
    if (op == 0) {
      len = 6;  // write32
    } else if (op == 3 || op == 4) {
      len = 7;  // maskwrite32, maskpoll32
    } else if (op == 1) {
      len = w[pos + 3] / 4;  // blockwrite
    } else if (op == 0x80 || op == 0x81) {
      len = w[pos + 1] / 4;  // tct, ddr patch
    } else {
      throw std::runtime_error("q4nx gather: unexpected instruction opcode " +
                               std::to_string(op));
    }
    if (len == 0) {
      throw std::runtime_error("q4nx gather: malformed instruction stream");
    }
    if (op == 0x81 && w[pos + 8] == arg) {
      sites.push_back(pos);
    }
    pos += len;
  }
  return sites;
}

std::uint64_t PatchOffset(const std::vector<std::uint32_t>& w,
                          std::size_t site) {
  return w[site + 10] | (std::uint64_t{w[site + 11]} << 32);
}

void SetPatchOffset(std::vector<std::uint32_t>& w, std::size_t site,
                    std::uint64_t offset) {
  w[site + 10] = static_cast<std::uint32_t>(offset);
  w[site + 11] = static_cast<std::uint32_t>(offset >> 32);
}

// The design's placeholder scale words, SCALE_SENTINEL + round * TOKENS +
// token: NaN payloads that no other instruction word carries.
constexpr std::uint32_t kScaleSentinel = 0x7FA50000;
// The weight argument of the design's (X, W, Y, R).
constexpr std::uint32_t kWeightArg = 1;
constexpr std::size_t kArrayCores = q4nx::kCols * q4nx::kRows;
// Output rounds of a down gather (kGemvHidden rows).
constexpr std::size_t kDownRounds = kGemvHidden / q4nx::kOutPerRound;

std::string GatherSequence(Q4nxGather::Kind kind, std::size_t tokens) {
  const std::string experts = std::to_string(kGatherExperts);
  const std::string t = "_t" + std::to_string(tokens);
  if (kind == Q4nxGather::Kind::kGateUp) {
    return "q4nx_gemv_k" + std::to_string(kGemvHidden) + "_n" +
           std::to_string(kGatherExperts * q4nx::kOutPerRound) +
           "_resnorm_swiglu_scaled_rounds" + experts + t;
  }
  return "q4nx_gemv_k" + std::to_string(kGatherExperts * kExpertFfn) + "_n" +
         std::to_string(kGemvHidden) + "_act_store_k" + experts + t;
}

}  // namespace

struct Q4nxGather::Impl {
  GemvRuntime::Impl& rt;
  Kind kind;
  ExpertArena arena;
  std::size_t tokens;
  std::unique_ptr<InstrSequence> instructions;
  // A weight patch: the expert slot (index into Select's ids) it reads and
  // its offset inside that expert's matrix.
  struct Site {
    std::size_t word;
    std::size_t expert;
    std::uint64_t offset;
  };
  std::vector<Site> sites;
  // A scale word (gate + up): its expert slot and token.
  struct Scale {
    std::size_t word;
    std::size_t expert;
    std::size_t token;
  };
  std::vector<Scale> scales;

  Impl(GemvRuntime::Impl& r, Kind k, const ExpertArena& a, std::size_t t)
      : rt(r), kind(k), arena(a), tokens(t) {
    if (rt.flow != Flow::kInstructionBuffer) {
      throw std::logic_error(
          "q4nx gather: needs a runtime with the instruction-buffer flow");
    }
    if (tokens < 1 || tokens > kGemvTokens) {
      throw std::invalid_argument("q4nx gather: 1 or 2 tokens per call");
    }
    const std::string name = GatherSequence(kind, tokens);
    instructions = std::make_unique<InstrSequence>(rt.hw, *rt.kernel,
                                                   rt.Path(name, ".insts.bin"));
    const auto& w = instructions->words;
    // The compiled offsets place segments back to back in a virtual matrix:
    //   gate + up: segment round * kCols + col, a column's slice of the
    //     gate + up of the round's expert;
    //   down: segment (round * kCols + col) * kGatherExperts + expert, the
    //     expert's part of that (round, col) slice.
    const std::size_t seg =
        kind == Kind::kGateUp
            ? ExpertArena::kUpBytes / q4nx::kCols
            : ExpertArena::kDownBytes / (kDownRounds * q4nx::kCols);
    for (const std::size_t word : PatchSites(w, kWeightArg)) {
      const std::uint64_t placeholder = PatchOffset(w, word);
      if (placeholder % seg != 0) {
        throw std::runtime_error("q4nx gather: unexpected patch offset in " +
                                 name);
      }
      const std::uint64_t index = placeholder / seg;
      const Site site =
          kind == Kind::kGateUp
              ? Site{word, index / q4nx::kCols, (index % q4nx::kCols) * seg}
              : Site{word, index % kGatherExperts,
                     (index / kGatherExperts) * seg};
      if (site.expert >= kGatherExperts) {
        throw std::runtime_error("q4nx gather: patch beyond the experts in " +
                                 name);
      }
      sites.push_back(site);
    }
    const std::size_t want_sites =
        kGatherExperts * q4nx::kCols *
        (kind == Kind::kGateUp ? std::size_t{1} : kDownRounds);
    if (sites.size() != want_sites) {
      throw std::runtime_error("q4nx gather: " + std::to_string(sites.size()) +
                               " weight patches in " + name + ", expected " +
                               std::to_string(want_sites));
    }
    if (kind != Kind::kGateUp) {
      return;
    }
    for (std::size_t i = 0; i < w.size(); ++i) {
      if (w[i] >= kScaleSentinel &&
          w[i] < kScaleSentinel + kGatherExperts * kGemvTokens) {
        const std::size_t j = w[i] - kScaleSentinel;
        scales.push_back({i, j / kGemvTokens, j % kGemvTokens});
      }
    }
    const std::size_t want_scales = kArrayCores * kGatherExperts * kGemvTokens;
    if (scales.size() != want_scales) {
      throw std::runtime_error("q4nx gather: " + std::to_string(scales.size()) +
                               " scale words in " + name + ", expected " +
                               std::to_string(want_scales));
    }
  }
};

Q4nxGather::Q4nxGather(GemvRuntime& runtime, Kind kind,
                       const ExpertArena& arena, std::size_t tokens)
    : impl_(std::make_unique<Impl>(runtime.impl(), kind, arena, tokens)) {}

Q4nxGather::~Q4nxGather() = default;

void Q4nxGather::Select(const int* ids, const float* scales) {
  Impl& m = *impl_;
  auto& w = m.instructions->words;
  for (std::size_t i = 0; i < kGatherExperts; ++i) {
    if (ids[i] < 0 || static_cast<std::size_t>(ids[i]) >= m.arena.experts) {
      throw std::out_of_range("q4nx gather: expert outside the arena");
    }
  }
  for (const auto& site : m.sites) {
    const auto e = static_cast<std::size_t>(ids[site.expert]);
    const std::size_t base =
        m.kind == Kind::kGateUp ? m.arena.UpOffset(e) : m.arena.DownOffset(e);
    SetPatchOffset(w, site.word, base + site.offset);
  }
  if (m.kind == Kind::kGateUp) {
    if (scales == nullptr) {
      throw std::invalid_argument("q4nx gather: gate + up needs scales");
    }
    for (const auto& s : m.scales) {
      // A token past the call's count has no output; its scale is unused.
      const float v =
          s.token < m.tokens ? scales[s.expert * m.tokens + s.token] : 0.0f;
      std::memcpy(&w[s.word], &v, sizeof(v));
    }
  }
  m.instructions->Sync();
}

std::size_t Q4nxGather::y_bytes() const {
  if (impl_->kind == Kind::kGateUp) {
    // As a SwiGLU Q4nxGemv: compact bf16 tokens kGemvMaxDim apart, and the
    // padding the design discards past them.
    return 8 * (kGemvMaxDim / 2) * sizeof(float);
  }
  return ((kGemvTokens - 1) * kGemvSlot + kGemvHidden) * sizeof(float);
}

std::size_t Q4nxGather::tokens() const {
  return impl_->tokens;
}

struct Chain::Impl {
  xrt::runlist runlist;
  std::vector<xrt::run> runs;
};

Chain::Chain(GemvRuntime& runtime)
    : impl_(std::make_unique<Impl>(
          Impl{xrt::runlist(*runtime.impl().hw.context), {}})) {}

Chain::~Chain() = default;

void Chain::Add(Q4nxGemv& gemv, Buffer& x, Buffer& y, Buffer& residual,
                std::size_t tokens) {
  impl_->runs.push_back(gemv.impl().Bind(x, y, residual, tokens));
  impl_->runlist.add(impl_->runs.back());
}

void Chain::Add(Q4nxGather& gather, Buffer& x, Buffer& arena, Buffer& y,
                Buffer& residual) {
  auto& g = gather.impl();
  if (arena.size() < g.arena.bytes()) {
    throw std::invalid_argument("q4nx gather: arena buffer too small");
  }
  impl_->runs.push_back(g.instructions->Run(
      *g.rt.kernel,
      {&x.impl().bo, &arena.impl().bo, &y.impl().bo, &residual.impl().bo}));
  impl_->runlist.add(impl_->runs.back());
}

void Chain::Run() {
  impl_->runlist.execute();
  impl_->runlist.wait();
}

}  // namespace gufo_npu::npu
