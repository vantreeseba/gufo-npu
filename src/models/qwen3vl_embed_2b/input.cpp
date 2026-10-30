#include "models/qwen3vl_embed_2b/input.hpp"

#include <unicode/uchar.h>

#include <stdexcept>

namespace gufo_npu::qwen3vl_embed_2b {
namespace {

constexpr char kWhitespace[] = " \t\n\r\f\v";

// The last code point of non-empty UTF-8 `s` (invalid bytes as themselves).
UChar32 LastCodePoint(std::string_view s) {
  std::size_t start = s.size() - 1;
  while (start > 0 && (static_cast<unsigned char>(s[start]) & 0xC0) == 0x80) {
    --start;
  }
  const auto lead = static_cast<unsigned char>(s[start]);
  int length = 1;
  UChar32 cp = lead;
  if (lead >= 0xF0) {
    length = 4;
    cp = lead & 0x07;
  } else if (lead >= 0xE0) {
    length = 3;
    cp = lead & 0x0F;
  } else if (lead >= 0xC0) {
    length = 2;
    cp = lead & 0x1F;
  }
  if (start + static_cast<std::size_t>(length) != s.size()) {
    return lead;  // truncated sequence
  }
  for (int k = 1; k < length; ++k) {
    cp = (cp << 6) | (static_cast<unsigned char>(s[start + k]) & 0x3F);
  }
  return cp;
}

int Special(const Tokenizer& tokenizer, std::string_view token) {
  const int id = tokenizer.Find(token);
  if (id < 0) {
    throw std::runtime_error("qwen3vl: the vocabulary has no " +
                             std::string(token));
  }
  return id;
}

}  // namespace

std::string NormalizeInstruction(std::string_view instruction) {
  const auto first = instruction.find_first_not_of(kWhitespace);
  if (first == std::string_view::npos) {
    return kDefaultInstruction;
  }
  const auto last = instruction.find_last_not_of(kWhitespace);
  std::string out(instruction.substr(first, last - first + 1));
  if (u_ispunct(LastCodePoint(out)) == 0) {
    out += '.';
  }
  return out;
}

std::string FormatInput(std::string_view text, std::string_view instruction) {
  std::string out = "<|im_start|>system\n";
  out += NormalizeInstruction(instruction);
  out += "<|im_end|>\n<|im_start|>user\n";
  out += text.empty() ? std::string_view(kEmptyText) : text;
  out += "<|im_end|>\n<|im_start|>assistant\n";
  return out;
}

std::vector<int> InputIds(const Tokenizer& tokenizer, std::string_view text,
                          std::string_view instruction) {
  const int end = Special(tokenizer, "<|endoftext|>");
  Special(tokenizer, "<|im_start|>");
  Special(tokenizer, "<|im_end|>");
  std::vector<int> ids = tokenizer.Encode(FormatInput(text, instruction));
  ids.push_back(end);
  return ids;
}

std::vector<int> InputIdsFromTokens(const Tokenizer& tokenizer,
                                    const std::vector<int>& content,
                                    std::string_view instruction) {
  if (content.empty()) {
    return InputIds(tokenizer, "", instruction);
  }
  const int end = Special(tokenizer, "<|endoftext|>");
  Special(tokenizer, "<|im_start|>");
  Special(tokenizer, "<|im_end|>");
  std::vector<int> ids = tokenizer.Encode("<|im_start|>system\n" +
                                          NormalizeInstruction(instruction) +
                                          "<|im_end|>\n<|im_start|>user\n");
  ids.insert(ids.end(), content.begin(), content.end());
  const auto tail = tokenizer.Encode("<|im_end|>\n<|im_start|>assistant\n");
  ids.insert(ids.end(), tail.begin(), tail.end());
  ids.push_back(end);
  return ids;
}

std::vector<int> ImageInputIds(const Tokenizer& tokenizer, int image_tokens,
                               std::string_view text,
                               std::string_view instruction) {
  if (image_tokens < 1) {
    throw std::invalid_argument("qwen3vl: an image needs at least one token");
  }
  const int end = Special(tokenizer, "<|endoftext|>");
  const int pad = Special(tokenizer, kImagePad);
  Special(tokenizer, "<|im_start|>");
  Special(tokenizer, "<|im_end|>");
  Special(tokenizer, "<|vision_start|>");
  Special(tokenizer, "<|vision_end|>");
  std::vector<int> ids = tokenizer.Encode(
      "<|im_start|>system\n" + NormalizeInstruction(instruction) +
      "<|im_end|>\n<|im_start|>user\n<|vision_start|>");
  ids.insert(ids.end(), static_cast<std::size_t>(image_tokens), pad);
  std::string tail = "<|vision_end|>";
  tail += text;
  tail += "<|im_end|>\n<|im_start|>assistant\n";
  const auto rest = tokenizer.Encode(tail);
  ids.insert(ids.end(), rest.begin(), rest.end());
  ids.push_back(end);
  return ids;
}

}  // namespace gufo_npu::qwen3vl_embed_2b
