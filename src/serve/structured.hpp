#pragma once

#include <cstdint>
#include <memory>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>
#include <vector>

#include "core/sampling.hpp"

// Structured outputs: grammars for a JSON value (any object, or one that
// matches a JSON schema) and for a forced Qwen3.5 tool call, and a
// TokenConstraint that keeps generated tokens inside one of them.
//
// The grammar is matched byte by byte with a stack machine; at each step the
// tokens that may come next are found by walking a trie of the vocabulary's
// bytes from the current state, pruning at the first byte the grammar
// refuses. Inside strings and free text, where nearly every token fits,
// the tokens that cannot end the text are allowed from a precomputed list
// and only the rest are walked.
namespace gufo_npu::serve {

// A schema or tool list the grammar compiler cannot take.
struct GrammarError : std::invalid_argument {
  using std::invalid_argument::invalid_argument;
};

// The vocabulary as the constraint walks it. Built once per model.
class VocabIndex {
public:
  // `bytes[id]`: the token's bytes (empty: never allowed); `special[id]`:
  // an added token such as "<tool_call>", allowed only where the grammar
  // spells out markup, never inside strings or free text.
  VocabIndex(std::vector<std::string> bytes,
             std::vector<std::uint8_t> special);
  ~VocabIndex();
  VocabIndex(const VocabIndex&) = delete;
  VocabIndex& operator=(const VocabIndex&) = delete;

  int size() const;

  struct Data;
  const Data& data() const;

private:
  std::unique_ptr<Data> data_;
};

class Grammar;

// Any JSON object (response_format {"type": "json_object"}).
std::shared_ptr<const Grammar> JsonObjectGrammar();

// A JSON value matching `schema` (response_format json_schema). With
// `strict`, objects without "additionalProperties" allow no other keys
// (OpenAI's strict mode); otherwise they allow any. Enforced: type,
// properties, required, additionalProperties, items, minItems, maxItems,
// minLength, maxLength, enum, const, anyOf, oneOf (as anyOf), allOf with
// one schema, $ref to "#", "#/$defs/..." or "#/definitions/...". Other
// keywords (pattern, format, minimum, ...) are not enforced.
std::shared_ptr<const Grammar> JsonSchemaGrammar(
    const nlohmann::ordered_json& schema, bool strict);

// One or more Qwen3.5 tool calls (`<tool_call>\n<function=NAME>\n
// <parameter=KEY>\nVALUE\n</parameter>\n...</function>\n</tool_call>`) of
// the functions in `tools` (OpenAI tools), or only of `name` when it is not
// empty. Parameter values follow each parameter's schema: strings as raw
// text, string enums as one of their values, other types as JSON; required
// parameters must appear and only declared ones may (any, when a function
// declares none).
std::shared_ptr<const Grammar> ToolCallGrammar(
    const nlohmann::ordered_json& tools, const std::string& name);

// A constraint that keeps the output inside `grammar`. With `thinking`, the
// output is free until "</think>" and the grammar applies after it. The
// `end_ids` (e.g. "<|im_end|>") are allowed once the grammar is complete.
std::shared_ptr<TokenConstraint> NewGrammarConstraint(
    std::shared_ptr<const Grammar> grammar,
    std::shared_ptr<const VocabIndex> vocab, std::vector<int> end_ids,
    bool thinking);

}  // namespace gufo_npu::serve
