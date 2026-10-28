#include <stdexcept>

#include "npu/program.hpp"

namespace gufo_npu::npu {
namespace {
[[noreturn]] void NoXrt() {
  throw std::runtime_error(
      "npu program: built without XRT (GUFO_NPU_ENABLE_XRT=OFF)");
}
}  // namespace

struct Program::Impl {};
Program::Program(const std::filesystem::path&, const std::string&) {
  NoXrt();
}
Program::~Program() = default;
const std::string& Program::name() const {
  NoXrt();
}
bool Program::Has(const std::string&) const {
  return false;
}

Buffer::Buffer(Program&, std::size_t) {
  NoXrt();
}

struct Batch::Impl {};
Batch::Batch(Program&) {
  NoXrt();
}
Batch::~Batch() = default;
void Batch::Add(const std::string&, std::initializer_list<const Buffer*>) {
  NoXrt();
}
std::size_t Batch::size() const {
  return 0;
}
void Batch::Run() {}
void Batch::Clear() {}

}  // namespace gufo_npu::npu
