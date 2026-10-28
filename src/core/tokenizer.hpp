#pragma once

#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "core/gguf.hpp"

// Byte-level BPE tokenizer (GGUF `tokenizer.ggml.model = gpt2`) for Qwen3.5:
// added tokens are split out first, then NFC normalisation, the Qwen3.5
// pre-tokenizer regex, and BPE over each piece. Matches HF `tokenizers`.
namespace gufo_npu {

class Tokenizer {
public:
  // Reads tokens, merges and token types from the GGUF metadata.
  explicit Tokenizer(const gguf::File& file);
  ~Tokenizer();
  Tokenizer(const Tokenizer&) = delete;
  Tokenizer& operator=(const Tokenizer&) = delete;

  std::vector<int> Encode(std::string_view text) const;
  // Concatenated UTF-8 bytes of `ids` (control tokens as their text).
  std::string Decode(const std::vector<int>& ids) const;
  std::string TokenBytes(int id) const;

  // Id of an exact token string such as "<|im_end|>", or -1.
  int Find(std::string_view token) const;
  int VocabSize() const { return static_cast<int>(bytes_.size()); }
  // Whether `id` is an added token (control or user-defined, such as
  // "<|im_end|>" or "<tool_call>"), which TokenBytes gives as its text.
  bool IsSpecial(int id) const {
    return id >= 0 && id < VocabSize() &&
           special_[static_cast<std::size_t>(id)] != 0;
  }

private:
  void EncodePiece(const std::string& piece, std::vector<int>& out) const;
  void EncodeText(std::string_view text, std::vector<int>& out) const;

  struct Regex;
  std::unique_ptr<Regex> regex_;
  std::vector<std::string> tokens_;    // byte-level encoded token strings
  std::vector<std::string> bytes_;     // decoded UTF-8 bytes per token
  std::vector<std::uint8_t> special_;  // 1 for added tokens
  std::unordered_map<std::string, int> vocab_;
  std::unordered_map<std::string, int> merge_rank_;  // "left right" -> rank
  // Added tokens (control / user-defined), longest first, for splitting.
  std::vector<std::pair<std::string, int>> added_;
  std::array<bool, 256> added_first_byte_{};
};

}  // namespace gufo_npu
