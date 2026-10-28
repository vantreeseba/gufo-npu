#pragma once

#include <cstddef>
#include <filesystem>
#include <initializer_list>
#include <memory>
#include <string>

#include "npu/gemv.hpp"

// An NPU program: an xclbin (<dir>/<name>.xclbin) loaded into its own
// hardware context, and the instruction sequences built for it
// (<dir>/<sequence>.insts.elf), for designs whose calls take buffers as
// their only runtime arguments. Its calls go through a Batch: runs added in
// order and executed as XRT runlists. Any npu::Buffer can be an argument,
// including windows (Buffer::Sub) of buffers another runtime allocated, so a
// program can read, for example, the decoder's resident weights in place.
//
// Moving the NPU from one hardware context to another costs milliseconds
// (docs/research/NPU-MOE-PREFILL.md), so callers batch their calls per
// program.
namespace gufo_npu::npu {

class Program {
public:
  // Throws when the xclbin is missing or the NPU cannot be opened.
  Program(const std::filesystem::path& dir, const std::string& name);
  ~Program();
  Program(const Program&) = delete;
  Program& operator=(const Program&) = delete;

  const std::string& name() const;
  // Whether `sequence` was built (its .insts.elf exists).
  bool Has(const std::string& sequence) const;

  struct Impl;
  Impl& impl() { return *impl_; }

private:
  std::unique_ptr<Impl> impl_;
};

// Calls of one program, run in the order they were added.
class Batch {
public:
  explicit Batch(Program& program);
  ~Batch();
  Batch(const Batch&) = delete;
  Batch& operator=(const Batch&) = delete;

  // A call of `sequence` with `args` as its runtime arguments, in order.
  // The batch keeps its own references to the buffers. Throws when the
  // sequence was not built.
  void Add(const std::string& sequence,
           std::initializer_list<const Buffer*> args);
  std::size_t size() const;
  // Runs every call and waits for the last. The calls stay for another Run.
  void Run();
  void Clear();

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace gufo_npu::npu
