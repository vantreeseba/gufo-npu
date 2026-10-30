#pragma once

#include <string>
#include <string_view>
#include <vector>

#include "core/tokenizer.hpp"

// What Qwen3-VL-Embedding embeds for a text, as its reference embedder
// (Qwen3VLEmbedder, scripts/qwen3_vl_embedding.py in the model repo) builds
// it: the chat template with the instruction as the system message and the
// text as the user's, the assistant prompt, and the <|endoftext|> the
// tokenizer's post-processor appends. The embedding is that last token's
// final hidden state.
namespace gufo_npu::qwen3vl_embed_2b {

// The system message when a request gives no instruction.
inline constexpr char kDefaultInstruction[] = "Represent the user's input.";
// What an empty input embeds (the reference's placeholder).
inline constexpr char kEmptyText[] = "NULL";
// The reference embedder's default length limit, in tokens.
inline constexpr int kDefaultMaxLength = 8192;

// The instruction as the reference uses it: trimmed, with a '.' appended
// when it does not end in punctuation (any Unicode P* category); empty
// gives kDefaultInstruction.
std::string NormalizeInstruction(std::string_view instruction);

// The templated text before tokenization (without <|endoftext|>):
//   <|im_start|>system\n{instruction}<|im_end|>\n
//   <|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n
std::string FormatInput(std::string_view text, std::string_view instruction);

// Token ids of FormatInput plus <|endoftext|>. Throws std::runtime_error
// when the vocabulary lacks the template's special tokens.
std::vector<int> InputIds(const Tokenizer& tokenizer, std::string_view text,
                          std::string_view instruction);

// The same around content given as token ids (an OpenAI token-array input):
// the ids take the text's place as they are.
std::vector<int> InputIdsFromTokens(const Tokenizer& tokenizer,
                                    const std::vector<int>& content,
                                    std::string_view instruction);

// The image placeholder token (one per merged image token).
inline constexpr char kImagePad[] = "<|image_pad|>";

// Token ids of an input holding one image (`image_tokens` placeholders)
// and optional text, as the reference embedder orders them (image first):
//   <|im_start|>system\n{instruction}<|im_end|>\n<|im_start|>user\n
//   <|vision_start|><|image_pad|> x image_tokens<|vision_end|>{text}
//   <|im_end|>\n<|im_start|>assistant\n<|endoftext|>
// Throws std::runtime_error when the vocabulary lacks a special token and
// std::invalid_argument for image_tokens < 1.
std::vector<int> ImageInputIds(const Tokenizer& tokenizer, int image_tokens,
                               std::string_view text,
                               std::string_view instruction);

}  // namespace gufo_npu::qwen3vl_embed_2b
