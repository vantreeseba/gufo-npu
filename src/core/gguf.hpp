#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <map>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

// Read-only, memory-mapped GGUF v3 reader.
namespace gufo_npu::gguf {

// ggml tensor types this engine can load.
enum class GgmlType : std::uint32_t {
  kF32 = 0,
  kF16 = 1,
  kQ8_0 = 8,
  kQ4_K = 12,
  kQ5_K = 13,
  kQ6_K = 14,
  kBF16 = 30,
};

std::string_view TypeName(GgmlType type);

// One metadata value. Arrays hold their items in `array`.
struct Value {
  enum class Kind { kUInt, kInt, kFloat, kBool, kString, kArray };
  Kind kind = Kind::kUInt;
  std::uint64_t u = 0;
  std::int64_t i = 0;
  double f = 0;
  bool b = false;
  std::string s;
  std::vector<Value> array;
};

struct TensorInfo {
  std::string name;
  // ggml order: shape[0] is the contiguous (row) dimension.
  std::vector<std::uint64_t> shape;
  GgmlType type = GgmlType::kF32;
  std::uint64_t offset = 0;  // from the start of the data section
  std::uint64_t num_elements = 0;
  std::uint64_t size_bytes = 0;
  const std::uint8_t* data = nullptr;

  std::uint64_t Rows() const;
  std::uint64_t RowElements() const { return shape.empty() ? 1 : shape[0]; }
};

class File {
public:
  // Throws std::runtime_error on I/O errors or a malformed file.
  static std::unique_ptr<File> Open(const std::filesystem::path& path);
  // Parses an in-memory image (used by tests). `bytes` must outlive the File.
  static std::unique_ptr<File> FromBytes(const std::uint8_t* bytes,
                                         std::size_t size);
  ~File();
  File(const File&) = delete;
  File& operator=(const File&) = delete;

  const Value* Find(std::string_view key) const;
  // Typed getters; throw when the key is missing or has the wrong kind.
  std::uint64_t GetUInt(std::string_view key) const;
  double GetFloat(std::string_view key) const;
  std::string GetString(std::string_view key) const;
  std::vector<std::string> GetStrings(std::string_view key) const;

  const TensorInfo* FindTensor(std::string_view name) const;
  const TensorInfo& Tensor(std::string_view name) const;  // throws if absent
  const std::vector<TensorInfo>& Tensors() const { return tensors_; }
  const std::map<std::string, Value, std::less<>>& Metadata() const {
    return metadata_;
  }

private:
  File() = default;
  void Parse();

  const std::uint8_t* base_ = nullptr;
  std::size_t size_ = 0;
  bool mapped_ = false;
  std::map<std::string, Value, std::less<>> metadata_;
  std::vector<TensorInfo> tensors_;
  std::map<std::string, std::size_t, std::less<>> tensor_index_;
};

}  // namespace gufo_npu::gguf
