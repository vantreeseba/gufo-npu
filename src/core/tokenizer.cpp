#include "core/tokenizer.hpp"

#include <unicode/normalizer2.h>
#include <unicode/regex.h>
#include <unicode/unistr.h>

#include <algorithm>
#include <array>
#include <limits>
#include <stdexcept>

namespace gufo_npu {
namespace {

// HF tokenizer.json pre_tokenizer for Qwen3.5 (Split, Isolated).
constexpr char kQwen35Pattern[] =
    R"((?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+)";

// GGUF token types that are matched literally before BPE.
constexpr int kTypeControl = 3;
constexpr int kTypeUserDefined = 4;

void AppendUtf8(std::string& out, char32_t cp) {
  if (cp < 0x80) {
    out += static_cast<char>(cp);
  } else if (cp < 0x800) {
    out += static_cast<char>(0xC0 | (cp >> 6));
    out += static_cast<char>(0x80 | (cp & 0x3F));
  } else {
    out += static_cast<char>(0xE0 | (cp >> 12));
    out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
    out += static_cast<char>(0x80 | (cp & 0x3F));
  }
}

// GPT-2 bytes_to_unicode: printable bytes map to themselves, the rest to
// U+0100 upwards.
std::array<char32_t, 256> ByteToCodepoint() {
  std::array<char32_t, 256> table{};
  char32_t next = 256;
  for (int b = 0; b < 256; ++b) {
    const bool printable = (b >= 33 && b <= 126) || (b >= 161 && b <= 172) ||
                           (b >= 174 && b <= 255);
    table[b] = printable ? static_cast<char32_t>(b) : next++;
  }
  return table;
}

const std::array<char32_t, 256>& ByteTable() {
  static const auto table = ByteToCodepoint();
  return table;
}

// Splits a UTF-8 string into code points (assumes valid UTF-8).
std::vector<char32_t> Codepoints(std::string_view s) {
  std::vector<char32_t> out;
  for (std::size_t i = 0; i < s.size();) {
    const auto c = static_cast<unsigned char>(s[i]);
    int len = 1;
    char32_t cp = c;
    if (c >= 0xF0) {
      len = 4;
      cp = c & 0x07;
    } else if (c >= 0xE0) {
      len = 3;
      cp = c & 0x0F;
    } else if (c >= 0xC0) {
      len = 2;
      cp = c & 0x1F;
    }
    for (int k = 1; k < len && i + k < s.size(); ++k) {
      cp = (cp << 6) | (static_cast<unsigned char>(s[i + k]) & 0x3F);
    }
    out.push_back(cp);
    i += len;
  }
  return out;
}

}  // namespace

struct Tokenizer::Regex {
  std::unique_ptr<icu::RegexPattern> pattern;
  const icu::Normalizer2* nfc = nullptr;
};

Tokenizer::Tokenizer(const gguf::File& file)
    : regex_(std::make_unique<Regex>()) {
  if (file.GetString("tokenizer.ggml.model") != "gpt2") {
    throw std::runtime_error(
        "tokenizer: only byte-level BPE (gpt2) is supported");
  }
  const auto pre = file.GetString("tokenizer.ggml.pre");
  if (pre != "qwen35") {
    throw std::runtime_error("tokenizer: unsupported pre-tokenizer " + pre);
  }
  tokens_ = file.GetStrings("tokenizer.ggml.tokens");
  const auto merges = file.GetStrings("tokenizer.ggml.merges");
  const auto* types = file.Find("tokenizer.ggml.token_type");

  // Reverse byte table for decoding.
  std::unordered_map<char32_t, unsigned char> cp_to_byte;
  for (int b = 0; b < 256; ++b) {
    cp_to_byte[ByteTable()[b]] = static_cast<unsigned char>(b);
  }

  bytes_.resize(tokens_.size());
  vocab_.reserve(tokens_.size());
  for (std::size_t id = 0; id < tokens_.size(); ++id) {
    const auto& token = tokens_[id];
    vocab_.emplace(token, static_cast<int>(id));
    int type = 1;
    if (types != nullptr && id < types->array.size()) {
      const auto& value = types->array[id];
      type = static_cast<int>(value.kind == gguf::Value::Kind::kInt
                                  ? value.i
                                  : static_cast<std::int64_t>(value.u));
    }
    if (type == kTypeControl || type == kTypeUserDefined) {
      bytes_[id] = token;
      added_.emplace_back(token, static_cast<int>(id));
      continue;
    }
    std::string decoded;
    for (const char32_t cp : Codepoints(token)) {
      const auto it = cp_to_byte.find(cp);
      if (it == cp_to_byte.end()) {
        decoded.clear();
        AppendUtf8(decoded, cp);  // not byte-level; keep as is
        break;
      }
      decoded += static_cast<char>(it->second);
    }
    bytes_[id] = decoded;
  }
  std::sort(added_.begin(), added_.end(), [](const auto& a, const auto& b) {
    return a.first.size() > b.first.size();
  });
  for (const auto& [token, id] : added_) {
    if (!token.empty()) {
      added_first_byte_[static_cast<unsigned char>(token[0])] = true;
    }
  }

  merge_rank_.reserve(merges.size());
  for (std::size_t rank = 0; rank < merges.size(); ++rank) {
    merge_rank_.emplace(merges[rank], static_cast<int>(rank));
  }

  UErrorCode status = U_ZERO_ERROR;
  UParseError parse_error{};
  regex_->pattern.reset(icu::RegexPattern::compile(
      icu::UnicodeString::fromUTF8(kQwen35Pattern), 0, parse_error, status));
  regex_->nfc = icu::Normalizer2::getNFCInstance(status);
  if (U_FAILURE(status) || !regex_->pattern) {
    throw std::runtime_error(std::string("tokenizer: ICU setup failed: ") +
                             u_errorName(status));
  }
}

Tokenizer::~Tokenizer() = default;

int Tokenizer::Find(std::string_view token) const {
  const auto it = vocab_.find(std::string(token));
  return it == vocab_.end() ? -1 : it->second;
}

std::string Tokenizer::TokenBytes(int id) const {
  if (id < 0 || id >= VocabSize()) {
    throw std::out_of_range("tokenizer: token id out of range");
  }
  return bytes_[id];
}

std::string Tokenizer::Decode(const std::vector<int>& ids) const {
  std::string out;
  for (const int id : ids) {
    out += TokenBytes(id);
  }
  return out;
}

void Tokenizer::EncodePiece(const std::string& piece,
                            std::vector<int>& out) const {
  // Byte-level symbols.
  std::vector<std::string> symbols;
  for (const char ch : piece) {
    std::string symbol;
    AppendUtf8(symbol, ByteTable()[static_cast<unsigned char>(ch)]);
    symbols.push_back(std::move(symbol));
  }
  // GPT-2 BPE: repeatedly merge the lowest-ranked adjacent pair.
  while (symbols.size() > 1) {
    int best = std::numeric_limits<int>::max();
    std::size_t best_at = 0;
    for (std::size_t i = 0; i + 1 < symbols.size(); ++i) {
      const auto it = merge_rank_.find(symbols[i] + " " + symbols[i + 1]);
      if (it != merge_rank_.end() && it->second < best) {
        best = it->second;
        best_at = i;
      }
    }
    if (best == std::numeric_limits<int>::max()) {
      break;
    }
    const std::string left = symbols[best_at];
    const std::string right = symbols[best_at + 1];
    std::vector<std::string> merged;
    merged.reserve(symbols.size());
    for (std::size_t i = 0; i < symbols.size();) {
      if (i + 1 < symbols.size() && symbols[i] == left &&
          symbols[i + 1] == right) {
        merged.push_back(left + right);
        i += 2;
      } else {
        merged.push_back(symbols[i]);
        ++i;
      }
    }
    symbols = std::move(merged);
  }
  for (const auto& symbol : symbols) {
    const auto it = vocab_.find(symbol);
    if (it == vocab_.end()) {
      throw std::runtime_error("tokenizer: no token for BPE symbol");
    }
    out.push_back(it->second);
  }
}

void Tokenizer::EncodeText(std::string_view text, std::vector<int>& out) const {
  if (text.empty()) {
    return;
  }
  UErrorCode status = U_ZERO_ERROR;
  const icu::UnicodeString source = icu::UnicodeString::fromUTF8(
      icu::StringPiece(text.data(), static_cast<int32_t>(text.size())));
  const icu::UnicodeString normalized = regex_->nfc->normalize(source, status);
  std::unique_ptr<icu::RegexMatcher> matcher(
      regex_->pattern->matcher(normalized, status));
  if (U_FAILURE(status)) {
    throw std::runtime_error("tokenizer: ICU normalize/match failed");
  }
  auto emit = [&](int32_t begin, int32_t end) {
    if (begin >= end) {
      return;
    }
    std::string piece;
    normalized.tempSubStringBetween(begin, end).toUTF8String(piece);
    EncodePiece(piece, out);
  };
  int32_t last = 0;
  while (matcher->find(status) && U_SUCCESS(status)) {
    const int32_t begin = matcher->start(status);
    const int32_t end = matcher->end(status);
    emit(last, begin);  // "Isolated": unmatched gaps are pieces too
    emit(begin, end);
    last = end;
  }
  emit(last, normalized.length());
}

std::vector<int> Tokenizer::Encode(std::string_view text) const {
  std::vector<int> out;
  std::size_t pos = 0;
  std::size_t plain_start = 0;
  while (pos < text.size()) {
    bool matched = false;
    if (added_first_byte_[static_cast<unsigned char>(text[pos])]) {
      for (const auto& [token, id] : added_) {
        if (text.compare(pos, token.size(), token) == 0) {
          EncodeText(text.substr(plain_start, pos - plain_start), out);
          out.push_back(id);
          pos += token.size();
          plain_start = pos;
          matched = true;
          break;
        }
      }
    }
    if (!matched) {
      ++pos;
    }
  }
  EncodeText(text.substr(plain_start), out);
  return out;
}

}  // namespace gufo_npu
