#include "serve/structured.hpp"

#include <algorithm>
#include <climits>
#include <cstring>
#include <map>
#include <string_view>
#include <utility>

namespace gufo_npu::serve {
namespace {

using Json = nlohmann::ordered_json;

constexpr int kNoMax = INT_MAX;
constexpr int kMaxDepth = 32;      // nesting, root included
constexpr int kMaxStacks = 16;     // parallel parses (anyOf)
constexpr int kMaxWs = 48;         // structural whitespace bytes in a row
constexpr int kMaxDigits = 24;     // digits before the exponent
constexpr int kMaxIntDigits = 18;  // integers: always fit int64
constexpr int kMaxExpDigits = 2;   // exponent digits: within a double
constexpr int kMaxKey = 256;       // bytes in an object key
constexpr int kMaxMembers = 128;   // properties, enum values, functions
constexpr int kEnd = 256;          // end of output, not a byte

// A set of up to 128 indices (properties, literals, functions).
struct Mask {
  std::uint64_t w[2] = {0, 0};

  void Set(int i) { w[i >> 6] |= std::uint64_t{1} << (i & 63); }
  void Clear(int i) { w[i >> 6] &= ~(std::uint64_t{1} << (i & 63)); }
  bool Test(int i) const { return (w[i >> 6] >> (i & 63)) & 1; }
  bool Empty() const { return w[0] == 0 && w[1] == 0; }
  bool SubsetOf(const Mask& o) const {
    return (w[0] & ~o.w[0]) == 0 && (w[1] & ~o.w[1]) == 0;
  }
  static Mask First(int n) {
    Mask m;
    for (int i = 0; i < n; ++i) {
      m.Set(i);
    }
    return m;
  }
  Mask Minus(const Mask& o) const {
    Mask m;
    m.w[0] = w[0] & ~o.w[0];
    m.w[1] = w[1] & ~o.w[1];
    return m;
  }
};

bool IsWs(int c) {
  return c == ' ' || c == '\n' || c == '\t' || c == '\r';
}
bool IsDigit(int c) {
  return c >= '0' && c <= '9';
}
bool IsHex(int c) {
  return IsDigit(c) || (c >= 'a' && c <= 'f') || (c >= 'A' && c <= 'F');
}
// A byte that may follow '\\' in a JSON string.
bool IsEscape(int c) {
  return c > 0 && c < 128 && std::strchr("\"\\/bfnrtu", c) != nullptr;
}

// UTF-8 validation one byte at a time (RFC 3629: no overlong forms, no
// surrogates, nothing past U+10FFFF). `st` packs the continuation bytes
// still due (bits 0-7) and the next byte's range (bits 8-15 and 16-23);
// 0 between characters. False for a byte that cannot come next.
bool Utf8Step(std::int32_t& st, int c) {
  auto expect = [&](int count, int lo, int hi) {
    st = count | (lo << 8) | (hi << 16);
    return true;
  };
  const int due = st & 0xFF;
  if (due > 0) {
    if (c < ((st >> 8) & 0xFF) || c > ((st >> 16) & 0xFF)) {
      return false;
    }
    if (due == 1) {
      st = 0;
      return true;
    }
    return expect(due - 1, 0x80, 0xBF);
  }
  if (c < 0x80) {
    return true;
  }
  if (c >= 0xC2 && c <= 0xDF) {
    return expect(1, 0x80, 0xBF);
  }
  if (c == 0xE0) {
    return expect(2, 0xA0, 0xBF);
  }
  if (c == 0xED) {
    return expect(2, 0x80, 0x9F);
  }
  if (c >= 0xE1 && c <= 0xEF) {
    return expect(2, 0x80, 0xBF);
  }
  if (c == 0xF0) {
    return expect(3, 0x90, 0xBF);
  }
  if (c >= 0xF1 && c <= 0xF3) {
    return expect(3, 0x80, 0xBF);
  }
  if (c == 0xF4) {
    return expect(3, 0x80, 0x8F);
  }
  return false;
}

// Whether `bytes` is whole, valid UTF-8.
bool WholeUtf8(const std::string& bytes) {
  std::int32_t st = 0;
  for (const char ch : bytes) {
    if (!Utf8Step(st, static_cast<unsigned char>(ch))) {
      return false;
    }
  }
  return st == 0;
}

}  // namespace

// ---------------------------------------------------------------------------
// The compiled grammar.

class Grammar {
public:
  enum class Kind : std::uint8_t {
    kAnyOf,
    kObject,
    kArray,
    kString,
    kNumber,
    kInteger,
    kLiterals,
    kNever,
  };

  struct Node {
    Kind kind = Kind::kNever;
    std::vector<int> alts;           // kAnyOf
    std::vector<std::string> names;  // kObject: keys as written, no quotes
    std::vector<int> props;          // kObject: node per name
    Mask required;                   // kObject
    int additional = -1;             // kObject: node for other keys
    int items = -1;                  // kArray
    int min_items = 0;
    int max_items = kNoMax;
    int min_len = 0;  // kString, in characters
    int max_len = kNoMax;
    std::vector<std::string> literals;  // kLiterals: exact texts
  };

  struct Function {
    std::string name;
    std::vector<std::string> params;
    std::vector<int> nodes;  // value node per parameter; -1: raw text
    Mask required;
    bool free_keys = false;  // no declared parameters: any key, raw text
  };

  std::vector<Node> nodes;
  // Per node: the stack frames a value of it needs beyond its own to be
  // completed (kInfinite: none can be). The matcher opens an object, an
  // array, a key or an item only when that many frames still fit, so every
  // value it starts can be finished within kMaxDepth.
  std::vector<int> need;
  int root = -1;  // JSON grammars
  bool tool = false;
  std::vector<Function> functions;
  Mask allowed_functions;

  int Add(Node node) {
    nodes.push_back(std::move(node));
    return static_cast<int>(nodes.size()) - 1;
  }
};

namespace {

using Kind = Grammar::Kind;
using Node = Grammar::Node;

// Any JSON value; returns its node.
int AddAny(Grammar& g) {
  const int any = g.Add(Node{});
  Node object;
  object.kind = Kind::kObject;
  object.additional = any;
  Node array;
  array.kind = Kind::kArray;
  array.items = any;
  Node string;
  string.kind = Kind::kString;
  Node number;
  number.kind = Kind::kNumber;
  Node literals;
  literals.kind = Kind::kLiterals;
  literals.literals = {"true", "false", "null"};
  std::vector<int> alts = {g.Add(object), g.Add(array), g.Add(string),
                           g.Add(number), g.Add(literals)};
  g.nodes[static_cast<std::size_t>(any)].kind = Kind::kAnyOf;
  g.nodes[static_cast<std::size_t>(any)].alts = std::move(alts);
  return any;
}

// The text of `name` between the quotes of its JSON string.
std::string KeyText(const std::string& name) {
  const std::string quoted = Json(name).dump();
  return quoted.substr(1, quoted.size() - 2);
}

int IntField(const Json& schema, const char* key, int fallback) {
  const auto it = schema.find(key);
  if (it == schema.end()) {
    return fallback;
  }
  if (!it->is_number_integer() || it->get<std::int64_t>() < 0) {
    throw GrammarError(std::string("schema: ") + key +
                       " must be a non-negative integer");
  }
  return static_cast<int>(
      std::min<std::int64_t>(it->get<std::int64_t>(), kNoMax));
}

class SchemaCompiler {
public:
  SchemaCompiler(Grammar& g, const Json& root, bool strict)
      : g_(g), root_(root), strict_(strict) {}

  int Compile(const Json& schema, int depth = 0) {
    if (depth > 64) {
      throw GrammarError("schema: nested too deeply");
    }
    if (schema.is_boolean()) {
      if (schema.get<bool>()) {
        return Any();
      }
      return g_.Add(Node{});  // kNever
    }
    if (!schema.is_object()) {
      throw GrammarError("schema: a schema must be an object or a boolean");
    }
    if (const auto ref = schema.find("$ref"); ref != schema.end()) {
      if (!ref->is_string()) {
        throw GrammarError("schema: $ref must be a string");
      }
      return Ref(ref->get<std::string>(), depth);
    }
    if (const auto it = schema.find("const"); it != schema.end()) {
      Node n;
      n.kind = Kind::kLiterals;
      n.literals = {it->dump()};
      return g_.Add(std::move(n));
    }
    if (const auto it = schema.find("enum"); it != schema.end()) {
      if (!it->is_array() || it->empty() ||
          it->size() > static_cast<std::size_t>(kMaxMembers)) {
        throw GrammarError("schema: enum must list 1 to 128 values");
      }
      Node n;
      n.kind = Kind::kLiterals;
      for (const auto& v : *it) {
        n.literals.push_back(v.dump());
      }
      return g_.Add(std::move(n));
    }
    for (const char* key : {"anyOf", "oneOf"}) {
      if (const auto it = schema.find(key); it != schema.end()) {
        if (!it->is_array() || it->empty()) {
          throw GrammarError(std::string("schema: ") + key +
                             " must be a non-empty array");
        }
        Node n;
        n.kind = Kind::kAnyOf;
        for (const auto& alt : *it) {
          n.alts.push_back(Compile(alt, depth + 1));
        }
        return g_.Add(std::move(n));
      }
    }
    if (const auto it = schema.find("allOf"); it != schema.end()) {
      if (!it->is_array() || it->size() != 1) {
        throw GrammarError("schema: allOf is supported with one schema only");
      }
      return Compile((*it)[0], depth + 1);
    }
    const auto type = schema.find("type");
    if (type != schema.end() && type->is_array()) {
      Node n;
      n.kind = Kind::kAnyOf;
      for (const auto& t : *type) {
        if (!t.is_string()) {
          throw GrammarError("schema: type must name types");
        }
        n.alts.push_back(OfType(t.get<std::string>(), schema, depth));
      }
      if (n.alts.empty()) {
        throw GrammarError("schema: type must name at least one type");
      }
      return g_.Add(std::move(n));
    }
    if (type != schema.end()) {
      if (!type->is_string()) {
        throw GrammarError("schema: type must be a string or strings");
      }
      return OfType(type->get<std::string>(), schema, depth);
    }
    // No type: inferred from the keywords, else any value.
    if (schema.contains("properties") ||
        schema.contains("additionalProperties") ||
        schema.contains("required")) {
      return OfType("object", schema, depth);
    }
    if (schema.contains("items") || schema.contains("minItems") ||
        schema.contains("maxItems")) {
      return OfType("array", schema, depth);
    }
    if (schema.contains("minLength") || schema.contains("maxLength")) {
      return OfType("string", schema, depth);
    }
    return Any();
  }

private:
  int Any() {
    if (any_ < 0) {
      any_ = AddAny(g_);
    }
    return any_;
  }

  int Ref(const std::string& ref, int depth) {
    if (const auto it = refs_.find(ref); it != refs_.end()) {
      return it->second;
    }
    const Json* target = nullptr;
    if (ref == "#") {
      target = &root_;
    } else {
      for (const char* prefix : {"#/$defs/", "#/definitions/"}) {
        const std::string_view p(prefix);
        if (ref.starts_with(p)) {
          const std::string name = ref.substr(p.size());
          const std::string defs(p.substr(2, p.size() - 3));
          const auto d = root_.find(defs);
          if (d != root_.end() && d->is_object() && d->contains(name)) {
            target = &(*d)[name];
          }
        }
      }
    }
    if (target == nullptr) {
      throw GrammarError("schema: cannot resolve $ref " + ref);
    }
    // A placeholder first, so recursive references find it.
    const int slot = g_.Add(Node{});
    refs_[ref] = slot;
    const int compiled = Compile(*target, depth + 1);
    Node n;
    n.kind = Kind::kAnyOf;
    n.alts = {compiled};
    g_.nodes[static_cast<std::size_t>(slot)] = std::move(n);
    return slot;
  }

  int OfType(const std::string& type, const Json& schema, int depth) {
    Node n;
    if (type == "string") {
      n.kind = Kind::kString;
      n.min_len = IntField(schema, "minLength", 0);
      n.max_len = IntField(schema, "maxLength", kNoMax);
      if (n.min_len > n.max_len) {
        throw GrammarError("schema: minLength is above maxLength");
      }
    } else if (type == "number") {
      n.kind = Kind::kNumber;
    } else if (type == "integer") {
      n.kind = Kind::kInteger;
    } else if (type == "boolean") {
      n.kind = Kind::kLiterals;
      n.literals = {"true", "false"};
    } else if (type == "null") {
      n.kind = Kind::kLiterals;
      n.literals = {"null"};
    } else if (type == "array") {
      n.kind = Kind::kArray;
      n.min_items = IntField(schema, "minItems", 0);
      n.max_items = IntField(schema, "maxItems", kNoMax);
      if (n.min_items > n.max_items) {
        throw GrammarError("schema: minItems is above maxItems");
      }
      const auto items = schema.find("items");
      n.items = items == schema.end() ? Any() : Compile(*items, depth + 1);
    } else if (type == "object") {
      return Object(schema, depth);
    } else {
      throw GrammarError("schema: unknown type " + type);
    }
    return g_.Add(std::move(n));
  }

  int Object(const Json& schema, int depth) {
    Node n;
    n.kind = Kind::kObject;
    if (const auto props = schema.find("properties"); props != schema.end()) {
      if (!props->is_object()) {
        throw GrammarError("schema: properties must be an object");
      }
      if (props->size() > static_cast<std::size_t>(kMaxMembers)) {
        throw GrammarError("schema: at most 128 properties per object");
      }
      for (const auto& [name, sub] : props->items()) {
        n.names.push_back(KeyText(name));
        n.props.push_back(Compile(sub, depth + 1));
      }
    }
    if (const auto req = schema.find("required"); req != schema.end()) {
      if (!req->is_array()) {
        throw GrammarError("schema: required must be an array");
      }
      for (const auto& r : *req) {
        if (!r.is_string()) {
          throw GrammarError("schema: required must list names");
        }
        const auto it = std::find(n.names.begin(), n.names.end(),
                                  KeyText(r.get<std::string>()));
        if (it == n.names.end()) {
          throw GrammarError("schema: required names a property \"" +
                             r.get<std::string>() +
                             "\" that properties does not define");
        }
        n.required.Set(static_cast<int>(it - n.names.begin()));
      }
    }
    const auto extra = schema.find("additionalProperties");
    if (extra == schema.end()) {
      n.additional = strict_ ? -1 : Any();
    } else if (extra->is_boolean()) {
      n.additional = extra->get<bool>() ? Any() : -1;
    } else {
      n.additional = Compile(*extra, depth + 1);
    }
    return g_.Add(std::move(n));
  }

  Grammar& g_;
  const Json& root_;
  bool strict_;
  int any_ = -1;
  std::map<std::string, int> refs_;
};

// A tool parameter's value: raw text for strings (and mixed types that
// include strings), one of the raw values for string enums, JSON otherwise.
int ParameterNode(Grammar& g, const Json& schema) {
  if (!schema.is_object()) {
    return -1;
  }
  for (const char* key : {"enum", "const"}) {
    const auto it = schema.find(key);
    if (it == schema.end()) {
      continue;
    }
    const Json values = it->is_array() ? *it : Json::array({*it});
    if (!values.empty() &&
        values.size() <= static_cast<std::size_t>(kMaxMembers) &&
        std::all_of(values.begin(), values.end(),
                    [](const Json& v) { return v.is_string(); })) {
      Node n;
      n.kind = Kind::kLiterals;
      for (const auto& v : values) {
        n.literals.push_back(v.get<std::string>());
      }
      return g.Add(std::move(n));
    }
  }
  std::vector<std::string> types;
  if (const auto t = schema.find("type"); t != schema.end()) {
    if (t->is_string()) {
      types.push_back(t->get<std::string>());
    } else if (t->is_array()) {
      for (const auto& v : *t) {
        if (v.is_string()) {
          types.push_back(v.get<std::string>());
        }
      }
    }
  }
  if (types.empty() ||
      std::find(types.begin(), types.end(), "string") != types.end()) {
    return -1;
  }
  SchemaCompiler compiler(g, schema, /*strict=*/false);
  return compiler.Compile(schema);
}

constexpr int kInfinite = INT_MAX / 4;

// Grammar::need, by iterating to a fixed point (recursive schemas).
void ComputeNeeds(Grammar& g) {
  g.need.assign(g.nodes.size(), kInfinite);
  for (bool changed = true; changed;) {
    changed = false;
    for (std::size_t i = 0; i < g.nodes.size(); ++i) {
      const Node& n = g.nodes[i];
      auto of = [&](int node) {
        return node < 0 ? 0 : g.need[static_cast<std::size_t>(node)];
      };
      int v = kInfinite;
      switch (n.kind) {
        case Kind::kNever:
          break;
        case Kind::kAnyOf:
          for (const int alt : n.alts) {
            v = std::min(v, of(alt));
          }
          break;
        case Kind::kObject:
          v = 0;
          for (std::size_t k = 0; k < n.props.size(); ++k) {
            if (n.required.Test(static_cast<int>(k))) {
              v = std::max(v, std::min(kInfinite, 1 + of(n.props[k])));
            }
          }
          break;
        case Kind::kArray:
          v = n.min_items > 0 ? std::min(kInfinite, 1 + of(n.items)) : 0;
          break;
        default:  // strings, numbers, literals replace their value frame
          v = 0;
          break;
      }
      if (v < g.need[i]) {
        g.need[i] = v;
        changed = true;
      }
    }
  }
}

// Checks that the root can be completed (the root frame and the value's).
void CheckRoot(const Grammar& g) {
  const int need = g.need[static_cast<std::size_t>(g.root)];
  if (need >= kInfinite) {
    throw GrammarError("schema: no finite JSON value matches it");
  }
  if (2 + need > kMaxDepth) {
    throw GrammarError("schema: its smallest value nests deeper than " +
                       std::to_string(kMaxDepth) + " levels");
  }
}

}  // namespace

std::shared_ptr<const Grammar> JsonObjectGrammar() {
  auto g = std::make_shared<Grammar>();
  const int any = AddAny(*g);
  Node object;
  object.kind = Kind::kObject;
  object.additional = any;
  g->root = g->Add(std::move(object));
  ComputeNeeds(*g);
  return g;
}

std::shared_ptr<const Grammar> JsonSchemaGrammar(const Json& schema,
                                                 bool strict) {
  auto g = std::make_shared<Grammar>();
  SchemaCompiler compiler(*g, schema, strict);
  g->root = compiler.Compile(schema);
  ComputeNeeds(*g);
  CheckRoot(*g);
  return g;
}

std::shared_ptr<const Grammar> ToolCallGrammar(const Json& tools,
                                               const std::string& name) {
  auto g = std::make_shared<Grammar>();
  g->tool = true;
  if (!tools.is_array() || tools.empty()) {
    throw GrammarError("tool_choice needs at least one tool");
  }
  if (tools.size() > static_cast<std::size_t>(kMaxMembers)) {
    throw GrammarError("tool_choice: at most 128 tools");
  }
  for (const auto& tool : tools) {
    const Json& f =
        tool.is_object() && tool.contains("function") ? tool["function"] : tool;
    if (!f.is_object() || !f.contains("name") || !f["name"].is_string()) {
      throw GrammarError("each tool needs a function name");
    }
    Grammar::Function fn;
    fn.name = f["name"].get<std::string>();
    if (fn.name.empty() || fn.name.find_first_of("<>\n") != std::string::npos) {
      throw GrammarError("tool name \"" + fn.name +
                         "\" cannot be forced (empty, or holds <, > or a "
                         "newline)");
    }
    const Json params =
        f.contains("parameters") ? f["parameters"] : Json::object();
    const auto props =
        params.is_object() ? params.find("properties") : params.end();
    if (params.is_object() && props != params.end() && props->is_object() &&
        !props->empty()) {
      if (props->size() > static_cast<std::size_t>(kMaxMembers)) {
        throw GrammarError("tool " + fn.name + ": at most 128 parameters");
      }
      for (const auto& [key, sub] : props->items()) {
        if (key.empty() || key.find_first_of("<>\n") != std::string::npos) {
          throw GrammarError("tool " + fn.name + ": parameter \"" + key +
                             "\" cannot be forced");
        }
        fn.params.push_back(key);
        fn.nodes.push_back(ParameterNode(*g, sub));
      }
      if (const auto req = params.find("required");
          req != params.end() && req->is_array()) {
        for (const auto& r : *req) {
          if (!r.is_string()) {
            continue;
          }
          const auto it = std::find(fn.params.begin(), fn.params.end(),
                                    r.get<std::string>());
          if (it != fn.params.end()) {
            fn.required.Set(static_cast<int>(it - fn.params.begin()));
          }
        }
      }
    } else {
      fn.free_keys = true;
    }
    g->functions.push_back(std::move(fn));
  }
  if (name.empty()) {
    g->allowed_functions = Mask::First(static_cast<int>(g->functions.size()));
  } else {
    for (std::size_t i = 0; i < g->functions.size(); ++i) {
      if (g->functions[i].name == name) {
        g->allowed_functions.Set(static_cast<int>(i));
      }
    }
    if (g->allowed_functions.Empty()) {
      throw GrammarError("tool_choice names \"" + name +
                         "\", which is not in tools");
    }
  }
  ComputeNeeds(*g);
  return g;
}

// ---------------------------------------------------------------------------
// The vocabulary.

namespace {

// Tokens by their bytes: first-child / next-sibling links.
struct Trie {
  struct TrieNode {
    int child = -1;
    int sibling = -1;
    int token = -1;  // first token ending here
    std::uint8_t byte = 0;
  };
  std::vector<TrieNode> nodes{TrieNode{}};

  void Insert(const std::string& bytes, int id, std::vector<int>& next_same) {
    int at = 0;
    for (const char ch : bytes) {
      const auto b = static_cast<std::uint8_t>(ch);
      int child = nodes[static_cast<std::size_t>(at)].child;
      int prev = -1;
      while (child >= 0 && nodes[static_cast<std::size_t>(child)].byte != b) {
        prev = child;
        child = nodes[static_cast<std::size_t>(child)].sibling;
      }
      if (child < 0) {
        TrieNode n;
        n.byte = b;
        nodes.push_back(n);
        child = static_cast<int>(nodes.size()) - 1;
        if (prev < 0) {
          nodes[static_cast<std::size_t>(at)].child = child;
        } else {
          nodes[static_cast<std::size_t>(prev)].sibling = child;
        }
      }
      at = child;
    }
    auto& end = nodes[static_cast<std::size_t>(at)];
    next_same[static_cast<std::size_t>(id)] = end.token;
    end.token = id;
  }
};

}  // namespace

struct VocabIndex::Data {
  std::vector<std::string> bytes;
  std::vector<std::uint8_t> special;
  std::vector<int> next_same;  // next token with the same bytes
  Trie all;                    // every ordinary token
  // Inside a free string, tokens of whole UTF-8 without '"', '\\' or a
  // control byte always fit (string_safe); the rest are walked.
  Trie string_unsafe;
  std::vector<int> string_safe;
  // Inside raw text, tokens of whole UTF-8 without '\n' or '<' always fit
  // (raw_safe): neither the end nor a forbidden tag can start in them.
  Trie raw_unsafe;
  std::vector<int> raw_safe;
  std::vector<int> specials;
};

VocabIndex::VocabIndex(std::vector<std::string> bytes,
                       std::vector<std::uint8_t> special)
    : data_(std::make_unique<Data>()) {
  Data& d = *data_;
  d.bytes = std::move(bytes);
  d.special = std::move(special);
  d.special.resize(d.bytes.size(), 0);
  d.next_same.assign(d.bytes.size(), -1);
  for (std::size_t i = 0; i < d.bytes.size(); ++i) {
    const int id = static_cast<int>(i);
    const std::string& b = d.bytes[i];
    if (b.empty()) {
      continue;
    }
    if (d.special[i] != 0) {
      d.specials.push_back(id);
      continue;
    }
    d.all.Insert(b, id, d.next_same);
    const bool whole = WholeUtf8(b);
    const bool string_safe =
        whole && std::all_of(b.begin(), b.end(), [](char ch) {
          const auto c = static_cast<unsigned char>(ch);
          return c >= 0x20 && c != '"' && c != '\\';
        });
    if (string_safe) {
      d.string_safe.push_back(id);
    } else {
      d.string_unsafe.Insert(b, id, d.next_same);
    }
    if (whole && b.find_first_of("\n<") == std::string::npos) {
      d.raw_safe.push_back(id);
    } else {
      d.raw_unsafe.Insert(b, id, d.next_same);
    }
  }
}

VocabIndex::~VocabIndex() = default;

int VocabIndex::size() const {
  return static_cast<int>(data_->bytes.size());
}

const VocabIndex::Data& VocabIndex::data() const {
  return *data_;
}

// ---------------------------------------------------------------------------
// The matcher.

namespace {

enum FrameKind : std::uint8_t {
  kValue,    // a value of `node` is next
  kObject,   // inside an object of `node`
  kArray,    // inside an array of `node`
  kString,   // inside a string of `node`
  kNumber,   // inside a number
  kLiteral,  // matching one of `node`'s literals
  kRoot,     // after the top-level value: whitespace, then the end
  kTool,     // a tool call's markup
  kRaw,      // a tool parameter's raw text, up to "\n</parameter>"
};

// kObject phases.
enum : std::uint8_t { kObjOpen, kObjKeyNext, kObjKey, kObjColon, kObjAfter };
// kArray phases.
enum : std::uint8_t { kArrOpen, kArrAfter, kArrNext };
// kNumber phases.
enum : std::uint8_t {
  kNumStart,
  kNumMinus,
  kNumZero,
  kNumInt,
  kNumDot,
  kNumFrac,
  kNumE,
  kNumESign,
  kNumExp,
};
// kTool phases.
enum : std::uint8_t {
  kToolLead,
  kToolOpen,
  kToolName,
  kToolNameNl,
  kToolItem,
  kToolKey,
  kToolKeyNl,
  kToolValueEnd,
  kToolAfter,
  kToolSep,
};

constexpr std::string_view kToolOpenText = "<tool_call>\n<function=";
constexpr std::string_view kParamOpen = "<parameter=";
constexpr std::string_view kCallClose = "</function>\n</tool_call>";
constexpr std::string_view kValueEnd = "\n</parameter>\n";
constexpr std::string_view kRawEnd = "\n</parameter>";
// Raw text may not hold these anywhere else: the chat format's parser ends
// a value at the first "</parameter>" and a call at the first
// "</tool_call>".
constexpr std::string_view kParamClose = "</parameter>";
constexpr std::string_view kToolCallClose = "</tool_call>";
constexpr std::string_view kThinkClose = "</think>";

struct Frame {
  std::uint8_t kind = kValue;
  std::uint8_t phase = 0;
  std::uint8_t esc = 0;    // strings: 0, 1 after '\\', 2..5 in \uXXXX
  std::uint8_t flags = 0;  // kValue: whitespace first; kNumber: integer
  std::int32_t node = -1;
  std::int32_t a = 0;     // counter: position, length, items, digits
  std::int32_t b = 0;     // kObject: the key's property; kTool: function
  std::int32_t c = 0;     // kTool: parameter
  std::int32_t utf8 = 0;  // strings, keys, raw text: Utf8Step's state
  Mask m1;                // kObject / kTool: seen; kLiteral: candidates
  Mask m2;                // kObject / kTool: key / name / item candidates
};

struct Stack {
  int depth = 0;
  int ws = 0;
  Frame f[kMaxDepth];
};

void CopyStack(Stack& dst, const Stack& src) {
  dst.depth = src.depth;
  dst.ws = src.ws;
  std::memcpy(static_cast<void*>(dst.f), static_cast<const void*>(src.f),
              sizeof(Frame) * static_cast<std::size_t>(src.depth));
}

bool SameStack(const Stack& x, const Stack& y) {
  return x.depth == y.depth && x.ws == y.ws &&
         std::memcmp(static_cast<const void*>(x.f),
                     static_cast<const void*>(y.f),
                     sizeof(Frame) * static_cast<std::size_t>(x.depth)) == 0;
}

// A set of parses; usually one.
struct StackSet {
  std::vector<Stack> stacks;
  int count = 0;

  void Clear() { count = 0; }
  Stack& Add() {
    if (static_cast<int>(stacks.size()) <= count) {
      stacks.resize(static_cast<std::size_t>(count) + 1);
    }
    return stacks[static_cast<std::size_t>(count++)];
  }
  void Push(const Stack& s) {
    for (int i = 0; i < count; ++i) {
      if (SameStack(stacks[static_cast<std::size_t>(i)], s)) {
        return;
      }
    }
    if (count >= kMaxStacks) {
      return;
    }
    CopyStack(Add(), s);
  }
  void Assign(const StackSet& o) {
    Clear();
    for (int i = 0; i < o.count; ++i) {
      CopyStack(Add(), o.stacks[static_cast<std::size_t>(i)]);
    }
  }
};

class Matcher {
public:
  explicit Matcher(const Grammar& g) : g_(g) {}

  void Initial(StackSet& out) const {
    out.Clear();
    Stack& s = out.Add();
    s.depth = 0;
    s.ws = 0;
    if (g_.tool) {
      Frame t;
      t.kind = kTool;
      t.phase = kToolLead;
      s.f[s.depth++] = t;
    } else {
      Frame r;
      r.kind = kRoot;
      s.f[s.depth++] = r;
      s.f[s.depth++] = ValueFrame(g_.root, true);
    }
  }

  // Every parse of `in` followed by byte `c` (or kEnd) into `out`.
  void Step(const StackSet& in, int c, bool special, StackSet& out) {
    out.Clear();
    for (int i = 0; i < in.count; ++i) {
      StepOne(in.stacks[static_cast<std::size_t>(i)], c, special, out);
    }
  }

  bool CanEnd(const StackSet& in) {
    Step(in, kEnd, false, end_scratch_);
    return end_scratch_.count > 0;
  }

private:
  enum class R { kAccept, kWs, kReject, kAgain, kFork };

  static Frame ValueFrame(int node, bool ws) {
    Frame f;
    f.kind = kValue;
    f.node = node;
    f.flags = ws ? 1 : 0;
    return f;
  }

  const Node& N(int i) const { return g_.nodes[static_cast<std::size_t>(i)]; }
  // Whether a value of node `i` started in a frame `above` frames over the
  // current top can still be completed within kMaxDepth.
  bool Fits(const Stack& s, int i, int above) const {
    return i < 0 ||
           s.depth + above + g_.need[static_cast<std::size_t>(i)] <= kMaxDepth;
  }

  Stack& Slot(int i) {
    if (static_cast<int>(work_.size()) <= i) {
      work_.resize(static_cast<std::size_t>(i) + 1);
    }
    return work_[static_cast<std::size_t>(i)];
  }

  // The parses of one stack after `c`; anyOf values fork into one parse
  // per alternative (at most kMaxForks per byte, so a schema whose
  // alternatives refer to themselves cannot loop).
  void StepOne(const Stack& start, int c, bool special, StackSet& out) {
    constexpr int kMaxForks = 256;
    int forks = 0;
    int top = 0;
    CopyStack(Slot(top++), start);
    while (top > 0) {
      const int i = --top;
      while (true) {
        Stack& s = work_[static_cast<std::size_t>(i)];
        if (s.depth == 0) {
          break;
        }
        const R r = Apply(s, c, special);
        if (r == R::kAgain) {
          continue;
        }
        if (r == R::kAccept || r == R::kWs) {
          if (r == R::kAccept) {
            s.ws = 0;
          }
          out.Push(s);
        } else if (r == R::kFork && ++forks <= kMaxForks) {
          CopyStack(fork_, s);  // slot i is reused below
          const Frame head = fork_.f[fork_.depth - 1];
          for (const int alt : N(head.node).alts) {
            Stack& d = Slot(top++);
            CopyStack(d, fork_);
            d.f[d.depth - 1] = ValueFrame(alt, head.flags != 0);
          }
        }
        break;
      }
    }
  }

  R Ws(Stack& s) {
    if (s.ws >= kMaxWs) {
      return R::kReject;
    }
    ++s.ws;
    return R::kWs;
  }

  static bool Push(Stack& s, const Frame& f) {
    if (s.depth >= kMaxDepth) {
      return false;
    }
    s.f[s.depth++] = f;
    return true;
  }

  R Apply(Stack& s, int c, bool special) {
    Frame& f = s.f[s.depth - 1];
    switch (f.kind) {
      case kValue:
        return ApplyValue(s, f, c);
      case kObject:
        return ApplyObject(s, f, c, special);
      case kArray:
        return ApplyArray(s, f, c);
      case kString:
        return ApplyString(s, f, c, special);
      case kNumber:
        return ApplyNumber(s, f, c);
      case kLiteral:
        return ApplyLiteral(s, f, c, special);
      case kRoot:
        if (c == kEnd) {
          return R::kAccept;
        }
        return IsWs(c) ? Ws(s) : R::kReject;
      case kTool:
        return ApplyTool(s, f, c, special);
      case kRaw:
        return ApplyRaw(s, f, c, special);
    }
    return R::kReject;
  }

  R ApplyValue(Stack& s, Frame& f, int c) {
    if (c == kEnd) {
      return R::kReject;
    }
    if (IsWs(c)) {
      return f.flags != 0 ? Ws(s) : R::kReject;
    }
    const Node& n = N(f.node);
    Frame next;
    switch (n.kind) {
      case Kind::kAnyOf:
        return R::kFork;
      case Kind::kObject:
        if (c != '{' || !Fits(s, f.node, 0)) {
          return R::kReject;
        }
        next.kind = kObject;
        next.node = f.node;
        next.phase = kObjOpen;
        f = next;
        return R::kAccept;
      case Kind::kArray:
        if (c != '[' || !Fits(s, f.node, 0)) {
          return R::kReject;
        }
        next.kind = kArray;
        next.node = f.node;
        next.phase = kArrOpen;
        f = next;
        return R::kAccept;
      case Kind::kString:
        if (c != '"') {
          return R::kReject;
        }
        next.kind = kString;
        next.node = f.node;
        f = next;
        return R::kAccept;
      case Kind::kNumber:
      case Kind::kInteger:
        if (c != '-' && !IsDigit(c)) {
          return R::kReject;
        }
        next.kind = kNumber;
        next.phase = kNumStart;
        next.flags = n.kind == Kind::kInteger ? 1 : 0;
        f = next;
        return R::kAgain;
      case Kind::kLiterals:
        next.kind = kLiteral;
        next.node = f.node;
        next.m1 = Mask::First(static_cast<int>(n.literals.size()));
        f = next;
        return R::kAgain;
      case Kind::kNever:
        return R::kReject;
    }
    return R::kReject;
  }

  // Starts a key at its opening quote; there must be room for its value.
  R BeginKey(const Stack& s, Frame& f) {
    if (s.depth >= kMaxDepth) {
      return R::kReject;
    }
    const Node& n = N(f.node);
    // Keys not seen yet whose value still fits; other keys only when an
    // additional value fits (f.c).
    f.m2 = Mask::First(static_cast<int>(n.names.size())).Minus(f.m1);
    for (std::size_t i = 0; i < n.names.size(); ++i) {
      if (f.m2.Test(static_cast<int>(i)) && !Fits(s, n.props[i], 1)) {
        f.m2.Clear(static_cast<int>(i));
      }
    }
    f.c = n.additional >= 0 && Fits(s, n.additional, 1) ? 1 : 0;
    if (f.m2.Empty() && f.c == 0) {
      return R::kReject;
    }
    f.phase = kObjKey;
    f.a = 0;
    f.esc = 0;
    f.utf8 = 0;
    return R::kAccept;
  }

  // Keeps the key candidates whose byte at f.a is `c`.
  R KeyByte(Frame& f, int c) {
    const Node& n = N(f.node);
    if (f.a >= kMaxKey) {
      return R::kReject;
    }
    Mask next;
    for (std::size_t i = 0; i < n.names.size(); ++i) {
      const int idx = static_cast<int>(i);
      if (f.m2.Test(idx) && n.names[i].size() > static_cast<std::size_t>(f.a) &&
          static_cast<unsigned char>(
              n.names[i][static_cast<std::size_t>(f.a)]) == c) {
        next.Set(idx);
      }
    }
    if (next.Empty() && f.c == 0) {
      return R::kReject;
    }
    f.m2 = next;
    ++f.a;
    return R::kAccept;
  }

  R ApplyObject(Stack& s, Frame& f, int c, bool special) {
    if (c == kEnd) {
      return R::kReject;
    }
    const Node& n = N(f.node);
    const bool complete = n.required.SubsetOf(f.m1);
    switch (f.phase) {
      case kObjOpen:
        if (IsWs(c)) {
          return Ws(s);
        }
        if (c == '"') {
          return BeginKey(s, f);
        }
        if (c == '}' && complete) {
          --s.depth;
          return R::kAccept;
        }
        return R::kReject;
      case kObjKeyNext:
        if (IsWs(c)) {
          return Ws(s);
        }
        return c == '"' ? BeginKey(s, f) : R::kReject;
      case kObjKey:
        if (special || c < 0x20 || !Utf8Step(f.utf8, c)) {
          return R::kReject;
        }
        if (f.esc == 1) {
          if (!IsEscape(c)) {
            return R::kReject;
          }
          f.esc = c == 'u' ? 2 : 0;
          return KeyByte(f, c);
        }
        if (f.esc >= 2) {
          if (!IsHex(c)) {
            return R::kReject;
          }
          f.esc = f.esc == 5 ? 0 : static_cast<std::uint8_t>(f.esc + 1);
          return KeyByte(f, c);
        }
        if (c == '"') {
          f.b = -1;
          for (std::size_t i = 0; i < n.names.size(); ++i) {
            if (f.m2.Test(static_cast<int>(i)) &&
                n.names[i].size() == static_cast<std::size_t>(f.a)) {
              f.b = static_cast<int>(i);
            }
          }
          if (f.b < 0 && f.c == 0) {
            return R::kReject;
          }
          f.phase = kObjColon;
          return R::kAccept;
        }
        if (c == '\\') {
          f.esc = 1;
        }
        return KeyByte(f, c);
      case kObjColon: {
        if (IsWs(c)) {
          return Ws(s);
        }
        if (c != ':') {
          return R::kReject;
        }
        int child = n.additional;
        if (f.b >= 0) {
          f.m1.Set(f.b);
          child = n.props[static_cast<std::size_t>(f.b)];
        }
        f.phase = kObjAfter;
        return Push(s, ValueFrame(child, true)) ? R::kAccept : R::kReject;
      }
      case kObjAfter:
        if (IsWs(c)) {
          return Ws(s);
        }
        if (c == ',') {
          // Another key only when one can still be completed.
          bool more = n.additional >= 0 && Fits(s, n.additional, 1);
          for (std::size_t i = 0; !more && i < n.names.size(); ++i) {
            more = !f.m1.Test(static_cast<int>(i)) && Fits(s, n.props[i], 1);
          }
          if (!more) {
            return R::kReject;
          }
          f.phase = kObjKeyNext;
          return R::kAccept;
        }
        if (c == '}' && complete) {
          --s.depth;
          return R::kAccept;
        }
        return R::kReject;
    }
    return R::kReject;
  }

  R ApplyArray(Stack& s, Frame& f, int c) {
    if (c == kEnd) {
      return R::kReject;
    }
    const Node& n = N(f.node);
    switch (f.phase) {
      case kArrOpen:
        if (IsWs(c)) {
          return Ws(s);
        }
        if (c == ']') {
          if (n.min_items > 0) {
            return R::kReject;
          }
          --s.depth;
          return R::kAccept;
        }
        if (n.max_items <= 0 || !Fits(s, n.items, 1)) {
          return R::kReject;
        }
        f.a = 1;
        f.phase = kArrAfter;
        return Push(s, ValueFrame(n.items, true)) ? R::kAgain : R::kReject;
      case kArrAfter:
        if (IsWs(c)) {
          return Ws(s);
        }
        if (c == ',' && f.a < n.max_items && Fits(s, n.items, 1)) {
          f.phase = kArrNext;
          return R::kAccept;
        }
        if (c == ']' && f.a >= n.min_items) {
          --s.depth;
          return R::kAccept;
        }
        return R::kReject;
      case kArrNext:
        if (IsWs(c)) {
          return Ws(s);
        }
        ++f.a;
        f.phase = kArrAfter;
        return Push(s, ValueFrame(n.items, true)) ? R::kAgain : R::kReject;
    }
    return R::kReject;
  }

  R ApplyString(Stack& s, Frame& f, int c, bool special) {
    if (c == kEnd || special) {
      return R::kReject;
    }
    const Node& n = N(f.node);
    if (f.esc == 1) {
      if (!IsEscape(c)) {
        return R::kReject;
      }
      f.esc = c == 'u' ? 2 : 0;
      return R::kAccept;
    }
    if (f.esc >= 2) {
      if (!IsHex(c)) {
        return R::kReject;
      }
      f.esc = f.esc == 5 ? 0 : static_cast<std::uint8_t>(f.esc + 1);
      return R::kAccept;
    }
    if (f.utf8 != 0) {  // inside a character: continuation bytes only
      return Utf8Step(f.utf8, c) ? R::kAccept : R::kReject;
    }
    if (c == '"') {
      if (f.a < n.min_len) {
        return R::kReject;
      }
      --s.depth;
      return R::kAccept;
    }
    if (c < 0x20 || !Utf8Step(f.utf8, c) || f.a >= n.max_len) {
      return R::kReject;
    }
    ++f.a;  // a character starts (or an escape)
    if (c == '\\') {
      f.esc = 1;
    }
    return R::kAccept;
  }

  R ApplyNumber(Stack& s, Frame& f, int c) {
    const bool integer = f.flags != 0;
    const bool can_end = f.phase == kNumZero || f.phase == kNumInt ||
                         f.phase == kNumFrac || f.phase == kNumExp;
    // a: digits before the exponent, b: exponent digits.
    auto digit = [&](std::uint8_t phase) {
      if (phase == kNumExp) {
        if (f.b >= kMaxExpDigits) {
          return R::kReject;
        }
        ++f.b;
      } else {
        if (f.a >= (integer ? kMaxIntDigits : kMaxDigits)) {
          return R::kReject;
        }
        ++f.a;
      }
      f.phase = phase;
      return R::kAccept;
    };
    auto end = [&] {
      if (!can_end) {
        return R::kReject;
      }
      --s.depth;
      return R::kAgain;  // the byte belongs to the parent
    };
    if (c == kEnd) {
      return end();
    }
    switch (f.phase) {
      case kNumStart:
        if (c == '-') {
          f.phase = kNumMinus;
          return R::kAccept;
        }
        [[fallthrough]];
      case kNumMinus:
        if (c == '0') {
          return digit(kNumZero);
        }
        if (IsDigit(c)) {
          return digit(kNumInt);
        }
        return R::kReject;
      case kNumZero:
      case kNumInt:
        if (f.phase == kNumInt && IsDigit(c)) {
          return digit(kNumInt);
        }
        if (!integer && c == '.') {
          f.phase = kNumDot;
          return R::kAccept;
        }
        if (!integer && (c == 'e' || c == 'E')) {
          f.phase = kNumE;
          return R::kAccept;
        }
        return end();
      case kNumDot:
        return IsDigit(c) ? digit(kNumFrac) : R::kReject;
      case kNumFrac:
        if (IsDigit(c)) {
          return digit(kNumFrac);
        }
        if (c == 'e' || c == 'E') {
          f.phase = kNumE;
          return R::kAccept;
        }
        return end();
      case kNumE:
        if (c == '+' || c == '-') {
          f.phase = kNumESign;
          return R::kAccept;
        }
        [[fallthrough]];
      case kNumESign:
        return IsDigit(c) ? digit(kNumExp) : R::kReject;
      case kNumExp:
        return IsDigit(c) ? digit(kNumExp) : end();
    }
    return R::kReject;
  }

  R ApplyLiteral(Stack& s, Frame& f, int c, bool special) {
    const Node& n = N(f.node);
    if (c != kEnd && !special) {
      Mask next;
      for (std::size_t i = 0; i < n.literals.size(); ++i) {
        const int idx = static_cast<int>(i);
        if (f.m1.Test(idx) &&
            n.literals[i].size() > static_cast<std::size_t>(f.a) &&
            static_cast<unsigned char>(
                n.literals[i][static_cast<std::size_t>(f.a)]) == c) {
          next.Set(idx);
        }
      }
      if (!next.Empty()) {
        f.m1 = next;
        ++f.a;
        return R::kAccept;
      }
    }
    for (std::size_t i = 0; i < n.literals.size(); ++i) {
      if (f.m1.Test(static_cast<int>(i)) &&
          n.literals[i].size() == static_cast<std::size_t>(f.a)) {
        --s.depth;
        return R::kAgain;
      }
    }
    return R::kReject;
  }

  const Grammar::Function& Fn(const Frame& f) const {
    return g_.functions[static_cast<std::size_t>(f.b)];
  }

  // The item options after a function's name or a parameter: bit 0 opens a
  // parameter, bit 1 closes the call.
  Mask ItemOptions(const Frame& f) const {
    const auto& fn = Fn(f);
    Mask m;
    if (fn.free_keys ||
        !Mask::First(static_cast<int>(fn.params.size())).Minus(f.m1).Empty()) {
      m.Set(0);
    }
    if (fn.required.SubsetOf(f.m1)) {
      m.Set(1);
    }
    return m;
  }

  R ApplyTool(Stack& s, Frame& f, int c, bool special) {
    switch (f.phase) {
      case kToolLead:
        if (c == kEnd) {
          return R::kReject;
        }
        if (c == '\n') {  // newlines only: the parser drops those
          return Ws(s);
        }
        f.phase = kToolOpen;
        f.a = 0;
        return R::kAgain;
      case kToolOpen:
        if (c == kEnd ||
            c != static_cast<unsigned char>(
                     kToolOpenText[static_cast<std::size_t>(f.a)])) {
          return R::kReject;
        }
        if (++f.a == static_cast<int>(kToolOpenText.size())) {
          f.phase = kToolName;
          f.a = 0;
          f.m2 = g_.allowed_functions;
        }
        return R::kAccept;
      case kToolName: {
        if (c == kEnd || special) {
          return R::kReject;
        }
        if (c == '>') {
          for (std::size_t i = 0; i < g_.functions.size(); ++i) {
            if (f.m2.Test(static_cast<int>(i)) &&
                g_.functions[i].name.size() == static_cast<std::size_t>(f.a)) {
              f.b = static_cast<int>(i);
              f.phase = kToolNameNl;
              return R::kAccept;
            }
          }
          return R::kReject;
        }
        Mask next;
        for (std::size_t i = 0; i < g_.functions.size(); ++i) {
          const auto& name = g_.functions[i].name;
          if (f.m2.Test(static_cast<int>(i)) &&
              name.size() > static_cast<std::size_t>(f.a) &&
              static_cast<unsigned char>(name[static_cast<std::size_t>(f.a)]) ==
                  c) {
            next.Set(static_cast<int>(i));
          }
        }
        if (next.Empty()) {
          return R::kReject;
        }
        f.m2 = next;
        ++f.a;
        return R::kAccept;
      }
      case kToolNameNl:
        if (c != '\n') {
          return R::kReject;
        }
        f.phase = kToolItem;
        f.a = 0;
        f.m1 = Mask{};
        f.m2 = ItemOptions(f);
        return R::kAccept;
      case kToolItem: {
        if (c == kEnd) {
          return R::kReject;
        }
        const std::string_view texts[2] = {kParamOpen, kCallClose};
        Mask next;
        for (int i = 0; i < 2; ++i) {
          if (f.m2.Test(i) && texts[i].size() > static_cast<std::size_t>(f.a) &&
              static_cast<unsigned char>(
                  texts[i][static_cast<std::size_t>(f.a)]) == c) {
            next.Set(i);
          }
        }
        if (next.Empty()) {
          return R::kReject;
        }
        f.m2 = next;
        ++f.a;
        if (next.Test(0) && f.a == static_cast<int>(kParamOpen.size())) {
          f.phase = kToolKey;
          f.a = 0;
          f.utf8 = 0;
          const auto& fn = Fn(f);
          f.m2 = Mask::First(static_cast<int>(fn.params.size())).Minus(f.m1);
        } else if (next.Test(1) && f.a == static_cast<int>(kCallClose.size())) {
          f.phase = kToolAfter;
          f.a = 0;
        }
        return R::kAccept;
      }
      case kToolKey: {
        if (c == kEnd || special || c == '\n' || c == '<' ||
            !Utf8Step(f.utf8, c)) {
          return R::kReject;
        }
        const auto& fn = Fn(f);
        if (c == '>') {
          if (f.a == 0) {
            return R::kReject;
          }
          f.c = -1;
          for (std::size_t i = 0; i < fn.params.size(); ++i) {
            if (f.m2.Test(static_cast<int>(i)) &&
                fn.params[i].size() == static_cast<std::size_t>(f.a)) {
              f.c = static_cast<int>(i);
            }
          }
          if (f.c < 0 && !fn.free_keys) {
            return R::kReject;
          }
          f.phase = kToolKeyNl;
          return R::kAccept;
        }
        if (f.a >= kMaxKey) {
          return R::kReject;
        }
        Mask next;
        for (std::size_t i = 0; i < fn.params.size(); ++i) {
          const auto& p = fn.params[i];
          if (f.m2.Test(static_cast<int>(i)) &&
              p.size() > static_cast<std::size_t>(f.a) &&
              static_cast<unsigned char>(p[static_cast<std::size_t>(f.a)]) ==
                  c) {
            next.Set(static_cast<int>(i));
          }
        }
        if (next.Empty() && !fn.free_keys) {
          return R::kReject;
        }
        f.m2 = next;
        ++f.a;
        return R::kAccept;
      }
      case kToolKeyNl: {
        if (c != '\n') {
          return R::kReject;
        }
        const auto& fn = Fn(f);
        int node = -1;
        if (f.c >= 0) {
          f.m1.Set(f.c);
          node = fn.nodes[static_cast<std::size_t>(f.c)];
        }
        f.phase = kToolValueEnd;
        Frame value;
        if (node < 0) {
          value.kind = kRaw;
          f.a = static_cast<int>(kRawEnd.size());  // the raw frame ends it
        } else {
          value = ValueFrame(node, false);
          f.a = 0;
        }
        return Push(s, value) ? R::kAccept : R::kReject;
      }
      case kToolValueEnd:
        if (c == kEnd || c != static_cast<unsigned char>(
                                  kValueEnd[static_cast<std::size_t>(f.a)])) {
          return R::kReject;
        }
        if (++f.a == static_cast<int>(kValueEnd.size())) {
          f.phase = kToolItem;
          f.a = 0;
          f.m2 = ItemOptions(f);
        }
        return R::kAccept;
      case kToolAfter:
        if (c == kEnd) {
          return R::kAccept;
        }
        if (c == '\n') {
          f.phase = kToolSep;
          return R::kAccept;
        }
        return R::kReject;
      case kToolSep:
        if (c == kEnd) {
          return R::kAccept;
        }
        if (c == '\n') {
          return Ws(s);
        }
        f.phase = kToolOpen;
        f.a = 0;
        return R::kAgain;
    }
    return R::kReject;
  }

  // Progress through `text` after byte `c`, from `at` bytes matched.
  // (Each text's first byte appears in it only once, so a mismatch
  // restarts at 0 or 1.)
  static int Track(std::string_view text, int at, int c) {
    if (c == static_cast<unsigned char>(text[static_cast<std::size_t>(at)])) {
      return at + 1;
    }
    return c == static_cast<unsigned char>(text[0]) ? 1 : 0;
  }

  // a: progress through "\n</parameter>" (the end), b: "</parameter>",
  // c: "</tool_call>".
  R ApplyRaw(Stack& s, Frame& f, int c, bool special) {
    if (c == kEnd || special || !Utf8Step(f.utf8, c)) {
      return R::kReject;
    }
    f.a = Track(kRawEnd, f.a, c);
    f.b = Track(kParamClose, f.b, c);
    f.c = Track(kToolCallClose, f.c, c);
    if (f.a == static_cast<int>(kRawEnd.size())) {
      --s.depth;
      return R::kAccept;
    }
    if (f.b == static_cast<int>(kParamClose.size()) ||
        f.c == static_cast<int>(kToolCallClose.size())) {
      return R::kReject;
    }
    return R::kAccept;
  }

  const Grammar& g_;
  std::vector<Stack> work_;
  Stack fork_;
  StackSet end_scratch_;
};

class GrammarConstraint : public TokenConstraint {
public:
  GrammarConstraint(std::shared_ptr<const Grammar> grammar,
                    std::shared_ptr<const VocabIndex> vocab,
                    std::vector<int> end_ids, bool thinking)
      : grammar_(std::move(grammar)),
        vocab_(std::move(vocab)),
        v_(vocab_->data()),
        end_ids_(std::move(end_ids)),
        thinking_(thinking),
        matcher_(*grammar_) {
    Reset();
  }

  void Reset() override {
    free_ = thinking_;
    dead_ = false;
    tail_.clear();
    matcher_.Initial(state_);
  }

  void Allowed(int vocab, std::vector<std::uint8_t>& mask) override {
    mask.assign(static_cast<std::size_t>(std::max(vocab, 0)), 0);
    const int n = std::min(vocab, static_cast<int>(v_.bytes.size()));
    if (free_) {
      for (int id = 0; id < n; ++id) {
        if (!v_.bytes[static_cast<std::size_t>(id)].empty() && !IsEnd(id)) {
          mask[static_cast<std::size_t>(id)] = 1;
        }
      }
      return;
    }
    if (dead_) {
      SetEnds(vocab, mask);
      return;
    }
    if (AllTops(kString, /*raw=*/false)) {
      for (const int id : v_.string_safe) {
        Set(mask, vocab, id);
      }
      Walk(v_.string_unsafe, mask, vocab);
    } else if (AllTops(kRaw, /*raw=*/true)) {
      for (const int id : v_.raw_safe) {
        Set(mask, vocab, id);
      }
      Walk(v_.raw_unsafe, mask, vocab);
    } else {
      Walk(v_.all, mask, vocab);
    }
    for (const int id : v_.specials) {
      if (id < vocab && !IsEnd(id) && Feed(state_, id, scratch_)) {
        mask[static_cast<std::size_t>(id)] = 1;
      }
    }
    if (matcher_.CanEnd(state_)) {
      SetEnds(vocab, mask);
    }
  }

  bool IsAllowed(int id) override {
    if (id < 0 || id >= static_cast<int>(v_.bytes.size())) {
      return false;
    }
    if (free_) {
      return !IsEnd(id) && !v_.bytes[static_cast<std::size_t>(id)].empty();
    }
    if (IsEnd(id)) {
      return dead_ || matcher_.CanEnd(state_);
    }
    if (dead_) {
      return false;
    }
    return Feed(state_, id, scratch_);
  }

  void Advance(int id) override {
    if (id < 0 || id >= static_cast<int>(v_.bytes.size()) || IsEnd(id)) {
      return;
    }
    const std::string& bytes = v_.bytes[static_cast<std::size_t>(id)];
    if (free_) {
      tail_ += bytes;
      const auto at = tail_.find(kThinkClose);
      if (at == std::string::npos) {
        if (tail_.size() > kThinkClose.size()) {
          tail_.erase(0, tail_.size() - kThinkClose.size());
        }
        return;
      }
      const std::string rest = tail_.substr(at + kThinkClose.size());
      free_ = false;
      tail_.clear();
      matcher_.Initial(state_);
      for (const char ch : rest) {
        matcher_.Step(state_, static_cast<unsigned char>(ch), false, scratch_);
        state_.Assign(scratch_);
      }
      dead_ = state_.count == 0;
      return;
    }
    if (dead_) {
      return;
    }
    if (Feed(state_, id, scratch_)) {
      state_.Assign(scratch_);
    } else {
      dead_ = true;  // a token the mask refused; only an end may follow
    }
  }

private:
  bool IsEnd(int id) const {
    return std::find(end_ids_.begin(), end_ids_.end(), id) != end_ids_.end();
  }

  void SetEnds(int vocab, std::vector<std::uint8_t>& mask) const {
    for (const int id : end_ids_) {
      if (id >= 0 && id < vocab) {
        mask[static_cast<std::size_t>(id)] = 1;
      }
    }
  }

  void Set(std::vector<std::uint8_t>& mask, int vocab, int id) const {
    if (id < vocab && !IsEnd(id)) {
      mask[static_cast<std::size_t>(id)] = 1;
    }
  }

  // Whether every parse is between characters inside a free string (no
  // length cap, not in an escape) or, with `raw`, inside raw text with no
  // end or forbidden tag under way.
  bool AllTops(std::uint8_t kind, bool raw) const {
    if (state_.count == 0) {
      return false;
    }
    for (int i = 0; i < state_.count; ++i) {
      const Stack& s = state_.stacks[static_cast<std::size_t>(i)];
      const Frame& top = s.f[s.depth - 1];
      if (top.kind != kind) {
        return false;
      }
      if (top.utf8 != 0) {
        return false;
      }
      if (raw) {
        // No end or forbidden tag under way.
        if (top.a != 0 || top.b != 0 || top.c != 0) {
          return false;
        }
      } else if (top.esc != 0 ||
                 grammar_->nodes[static_cast<std::size_t>(top.node)].max_len !=
                     kNoMax) {
        return false;
      }
    }
    return true;
  }

  // `state` followed by token `id`'s bytes, into `out`; false if refused.
  bool Feed(const StackSet& state, int id, StackSet& out) {
    const std::string& bytes = v_.bytes[static_cast<std::size_t>(id)];
    if (bytes.empty()) {
      return false;
    }
    const bool special = v_.special[static_cast<std::size_t>(id)] != 0;
    feed_.Assign(state);
    for (const char ch : bytes) {
      matcher_.Step(feed_, static_cast<unsigned char>(ch), special, out);
      if (out.count == 0) {
        return false;
      }
      feed_.Assign(out);
    }
    return true;
  }

  // Marks every token of `trie` the grammar accepts from state_.
  void Walk(const Trie& trie, std::vector<std::uint8_t>& mask, int vocab) {
    if (levels_.empty()) {
      levels_.resize(1);
    }
    levels_[0].Assign(state_);
    WalkFrom(trie, 0, 0, mask, vocab);
  }

  void WalkFrom(const Trie& trie, int node, std::size_t depth,
                std::vector<std::uint8_t>& mask, int vocab) {
    if (levels_.size() <= depth + 1) {
      levels_.resize(depth + 2);
    }
    for (int child = trie.nodes[static_cast<std::size_t>(node)].child;
         child >= 0;
         child = trie.nodes[static_cast<std::size_t>(child)].sibling) {
      const auto& tn = trie.nodes[static_cast<std::size_t>(child)];
      matcher_.Step(levels_[depth], tn.byte, false, levels_[depth + 1]);
      if (levels_[depth + 1].count == 0) {
        continue;
      }
      for (int id = tn.token; id >= 0;
           id = v_.next_same[static_cast<std::size_t>(id)]) {
        if (id < vocab && !IsEnd(id)) {
          mask[static_cast<std::size_t>(id)] = 1;
        }
      }
      if (tn.child >= 0) {
        WalkFrom(trie, child, depth + 1, mask, vocab);
      }
    }
  }

  std::shared_ptr<const Grammar> grammar_;
  std::shared_ptr<const VocabIndex> vocab_;
  const VocabIndex::Data& v_;
  std::vector<int> end_ids_;
  bool thinking_;
  Matcher matcher_;
  StackSet state_;
  StackSet scratch_;
  StackSet feed_;
  std::vector<StackSet> levels_;
  bool free_ = false;
  bool dead_ = false;
  std::string tail_;
};

}  // namespace

std::shared_ptr<TokenConstraint> NewGrammarConstraint(
    std::shared_ptr<const Grammar> grammar,
    std::shared_ptr<const VocabIndex> vocab, std::vector<int> end_ids,
    bool thinking) {
  if (!grammar || !vocab) {
    throw std::invalid_argument("NewGrammarConstraint: missing grammar");
  }
  return std::make_shared<GrammarConstraint>(
      std::move(grammar), std::move(vocab), std::move(end_ids), thinking);
}

}  // namespace gufo_npu::serve
