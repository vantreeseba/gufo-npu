#include "core/gguf.hpp"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cstring>
#include <stdexcept>

#include "core/quant.hpp"

namespace gufo_npu::gguf {
namespace {

constexpr std::uint32_t kMagic = 0x46554747;  // "GGUF" little-endian
constexpr std::uint64_t kDefaultAlignment = 32;

enum ValueType : std::uint32_t {
  kU8 = 0,
  kI8 = 1,
  kU16 = 2,
  kI16 = 3,
  kU32 = 4,
  kI32 = 5,
  kF32 = 6,
  kBool = 7,
  kString = 8,
  kArray = 9,
  kU64 = 10,
  kI64 = 11,
  kF64 = 12,
};

bool KnownType(std::uint32_t type) {
  switch (static_cast<GgmlType>(type)) {
    case GgmlType::kF32:
    case GgmlType::kF16:
    case GgmlType::kQ8_0:
    case GgmlType::kQ4_K:
    case GgmlType::kQ5_K:
    case GgmlType::kQ6_K:
    case GgmlType::kBF16:
      return true;
  }
  return false;
}

class Cursor {
public:
  Cursor(const std::uint8_t* data, std::size_t size)
      : data_(data), size_(size) {}

  template<typename T>
  T Read() {
    Need(sizeof(T));
    T value;
    std::memcpy(&value, data_ + pos_, sizeof(T));
    pos_ += sizeof(T);
    return value;
  }

  std::string ReadString() {
    const auto length = Read<std::uint64_t>();
    Need(length);
    std::string text(reinterpret_cast<const char*>(data_ + pos_), length);
    pos_ += length;
    return text;
  }

  std::size_t Position() const { return pos_; }

private:
  void Need(std::uint64_t n) const {
    if (n > size_ || pos_ > size_ - n) {
      throw std::runtime_error("GGUF: truncated header");
    }
  }

  const std::uint8_t* data_;
  std::size_t size_;
  std::size_t pos_ = 0;
};

Value ReadValue(Cursor& cursor, std::uint32_t type, int depth = 0) {
  Value value;
  switch (type) {
    case kU8:
      value.u = cursor.Read<std::uint8_t>();
      break;
    case kU16:
      value.u = cursor.Read<std::uint16_t>();
      break;
    case kU32:
      value.u = cursor.Read<std::uint32_t>();
      break;
    case kU64:
      value.u = cursor.Read<std::uint64_t>();
      break;
    case kI8:
      value.kind = Value::Kind::kInt;
      value.i = cursor.Read<std::int8_t>();
      break;
    case kI16:
      value.kind = Value::Kind::kInt;
      value.i = cursor.Read<std::int16_t>();
      break;
    case kI32:
      value.kind = Value::Kind::kInt;
      value.i = cursor.Read<std::int32_t>();
      break;
    case kI64:
      value.kind = Value::Kind::kInt;
      value.i = cursor.Read<std::int64_t>();
      break;
    case kF32:
      value.kind = Value::Kind::kFloat;
      value.f = cursor.Read<float>();
      break;
    case kF64:
      value.kind = Value::Kind::kFloat;
      value.f = cursor.Read<double>();
      break;
    case kBool:
      value.kind = Value::Kind::kBool;
      value.b = cursor.Read<std::uint8_t>() != 0;
      break;
    case kString:
      value.kind = Value::Kind::kString;
      value.s = cursor.ReadString();
      break;
    case kArray: {
      if (depth > 0) {
        throw std::runtime_error("GGUF: nested arrays are not supported");
      }
      value.kind = Value::Kind::kArray;
      const auto item_type = cursor.Read<std::uint32_t>();
      const auto count = cursor.Read<std::uint64_t>();
      value.array.reserve(count);
      for (std::uint64_t n = 0; n < count; ++n) {
        value.array.push_back(ReadValue(cursor, item_type, depth + 1));
      }
      break;
    }
    default:
      throw std::runtime_error("GGUF: unknown value type " +
                               std::to_string(type));
  }
  if (type == kU8 || type == kU16 || type == kU32 || type == kU64) {
    value.kind = Value::Kind::kUInt;
  }
  return value;
}

}  // namespace

std::string_view TypeName(GgmlType type) {
  switch (type) {
    case GgmlType::kF32:
      return "F32";
    case GgmlType::kF16:
      return "F16";
    case GgmlType::kQ8_0:
      return "Q8_0";
    case GgmlType::kQ4_K:
      return "Q4_K";
    case GgmlType::kQ5_K:
      return "Q5_K";
    case GgmlType::kQ6_K:
      return "Q6_K";
    case GgmlType::kBF16:
      return "BF16";
  }
  return "?";
}

std::uint64_t TensorInfo::Rows() const {
  std::uint64_t rows = 1;
  for (std::size_t d = 1; d < shape.size(); ++d) {
    rows *= shape[d];
  }
  return rows;
}

std::unique_ptr<File> File::Open(const std::filesystem::path& path) {
  const int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
  if (fd < 0) {
    throw std::runtime_error("GGUF: cannot open " + path.string());
  }
  struct stat info {};
  if (::fstat(fd, &info) != 0) {
    ::close(fd);
    throw std::runtime_error("GGUF: cannot stat " + path.string());
  }
  const auto size = static_cast<std::size_t>(info.st_size);
  void* map = ::mmap(nullptr, size, PROT_READ, MAP_PRIVATE, fd, 0);
  ::close(fd);
  if (map == MAP_FAILED) {
    throw std::runtime_error("GGUF: cannot mmap " + path.string());
  }
  std::unique_ptr<File> file(new File());
  file->base_ = static_cast<const std::uint8_t*>(map);
  file->size_ = size;
  file->mapped_ = true;
  file->Parse();
  return file;
}

std::unique_ptr<File> File::FromBytes(const std::uint8_t* bytes,
                                      std::size_t size) {
  std::unique_ptr<File> file(new File());
  file->base_ = bytes;
  file->size_ = size;
  file->Parse();
  return file;
}

File::~File() {
  if (mapped_) {
    ::munmap(const_cast<std::uint8_t*>(base_), size_);
  }
}

void File::Parse() {
  Cursor cursor(base_, size_);
  if (cursor.Read<std::uint32_t>() != kMagic) {
    throw std::runtime_error("GGUF: bad magic");
  }
  const auto version = cursor.Read<std::uint32_t>();
  if (version != 3) {
    throw std::runtime_error("GGUF: unsupported version " +
                             std::to_string(version));
  }
  const auto n_tensors = cursor.Read<std::uint64_t>();
  const auto n_kv = cursor.Read<std::uint64_t>();
  for (std::uint64_t n = 0; n < n_kv; ++n) {
    auto key = cursor.ReadString();
    const auto type = cursor.Read<std::uint32_t>();
    metadata_.emplace(std::move(key), ReadValue(cursor, type));
  }

  tensors_.reserve(n_tensors);
  for (std::uint64_t n = 0; n < n_tensors; ++n) {
    TensorInfo tensor;
    tensor.name = cursor.ReadString();
    const auto dims = cursor.Read<std::uint32_t>();
    if (dims == 0 || dims > 4) {
      throw std::runtime_error("GGUF: bad rank for " + tensor.name);
    }
    tensor.num_elements = 1;
    for (std::uint32_t d = 0; d < dims; ++d) {
      tensor.shape.push_back(cursor.Read<std::uint64_t>());
      tensor.num_elements *= tensor.shape.back();
    }
    const auto type = cursor.Read<std::uint32_t>();
    if (!KnownType(type)) {
      throw std::runtime_error("GGUF: unsupported ggml type " +
                               std::to_string(type) + " for " + tensor.name);
    }
    tensor.type = static_cast<GgmlType>(type);
    tensor.offset = cursor.Read<std::uint64_t>();
    if (tensor.shape[0] % quant::BlockElements(tensor.type) != 0) {
      throw std::runtime_error("GGUF: row of " + tensor.name +
                               " is not a whole number of blocks");
    }
    tensor.size_bytes =
        quant::RowBytes(tensor.type, tensor.shape[0]) * tensor.Rows();
    tensors_.push_back(std::move(tensor));
  }

  std::uint64_t alignment = kDefaultAlignment;
  if (const auto* value = Find("general.alignment")) {
    alignment = value->u;
  }
  const std::uint64_t data_start =
      (cursor.Position() + alignment - 1) / alignment * alignment;
  for (std::size_t n = 0; n < tensors_.size(); ++n) {
    auto& tensor = tensors_[n];
    const auto begin = data_start + tensor.offset;
    if (begin > size_ || tensor.size_bytes > size_ - begin) {
      throw std::runtime_error("GGUF: tensor " + tensor.name +
                               " runs past the end of the file");
    }
    tensor.data = base_ + begin;
    tensor_index_.emplace(tensor.name, n);
  }
}

const Value* File::Find(std::string_view key) const {
  const auto it = metadata_.find(key);
  return it == metadata_.end() ? nullptr : &it->second;
}

std::uint64_t File::GetUInt(std::string_view key) const {
  const auto* value = Find(key);
  if (value == nullptr) {
    throw std::runtime_error("GGUF: missing key " + std::string(key));
  }
  if (value->kind == Value::Kind::kUInt) {
    return value->u;
  }
  if (value->kind == Value::Kind::kInt && value->i >= 0) {
    return static_cast<std::uint64_t>(value->i);
  }
  throw std::runtime_error("GGUF: key " + std::string(key) +
                           " is not an unsigned integer");
}

double File::GetFloat(std::string_view key) const {
  const auto* value = Find(key);
  if (value == nullptr || value->kind != Value::Kind::kFloat) {
    throw std::runtime_error("GGUF: missing float key " + std::string(key));
  }
  return value->f;
}

std::string File::GetString(std::string_view key) const {
  const auto* value = Find(key);
  if (value == nullptr || value->kind != Value::Kind::kString) {
    throw std::runtime_error("GGUF: missing string key " + std::string(key));
  }
  return value->s;
}

std::vector<std::string> File::GetStrings(std::string_view key) const {
  const auto* value = Find(key);
  if (value == nullptr || value->kind != Value::Kind::kArray) {
    throw std::runtime_error("GGUF: missing array key " + std::string(key));
  }
  std::vector<std::string> out;
  out.reserve(value->array.size());
  for (const auto& item : value->array) {
    if (item.kind != Value::Kind::kString) {
      throw std::runtime_error("GGUF: key " + std::string(key) +
                               " is not a string array");
    }
    out.push_back(item.s);
  }
  return out;
}

const TensorInfo* File::FindTensor(std::string_view name) const {
  const auto it = tensor_index_.find(name);
  return it == tensor_index_.end() ? nullptr : &tensors_[it->second];
}

const TensorInfo& File::Tensor(std::string_view name) const {
  const auto* tensor = FindTensor(name);
  if (tensor == nullptr) {
    throw std::runtime_error("GGUF: missing tensor " + std::string(name));
  }
  return *tensor;
}

}  // namespace gufo_npu::gguf
