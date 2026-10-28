// The Qwen3.5 chat format: renders match the GGUF template (fixtures from
// tools/reference/qwen35_chat_template.py), and the output parser gives the
// same parts however the text is split across Feed calls.

#include "serve/chat_format.hpp"

#include <cstdio>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "models/qwen35/chat.hpp"

namespace {

using gufo_npu::serve::ChatPrompt;
using gufo_npu::serve::NewQwen35ChatFormat;
using gufo_npu::serve::OutputPart;
using Json = nlohmann::ordered_json;
using Kind = OutputPart::Kind;

int failures = 0;

void Fail(const std::string& what) {
  std::fprintf(stderr, "FAIL: %s\n", what.c_str());
  ++failures;
}

Json ReadJson(const std::string& path) {
  std::ifstream in(path);
  if (!in) {
    throw std::runtime_error("cannot read " + path);
  }
  return Json::parse(in);
}

std::string Show(const std::string& s) {
  std::string out;
  for (const char c : s) {
    out += c == '\n' ? std::string("\\n") : std::string(1, c);
  }
  return out;
}

void CheckRender(const std::string& dir) {
  const Json cases = ReadJson(dir + "/chat_cases.json");
  const Json expected = ReadJson(dir + "/chat_expected.json");
  const auto format = NewQwen35ChatFormat();
  int checked = 0;
  for (const auto& c : cases) {
    const std::string name = c["name"];
    ChatPrompt prompt;
    prompt.messages = c["messages"];
    prompt.tools = c.value("tools", Json::array());
    prompt.thinking = c.value("thinking", false);
    const Json& want = expected.at(name);
    try {
      const std::string got = format->Render(prompt);
      if (want.is_null()) {
        Fail("render " + name + ": expected a rejection");
      } else if (got != want.get<std::string>()) {
        const std::string w = want.get<std::string>();
        std::size_t at = 0;
        while (at < got.size() && at < w.size() && got[at] == w[at]) {
          ++at;
        }
        Fail("render " + name + " differs at byte " + std::to_string(at) +
             ":\n  got  ..." + Show(got.substr(at, 60)) + "\n  want ..." +
             Show(w.substr(at, 60)));
      }
    } catch (const std::invalid_argument& e) {
      if (!want.is_null()) {
        Fail("render " + name + ": unexpected rejection: " + e.what());
      }
    }
    ++checked;
  }
  std::printf("ok: %d render cases against the GGUF template\n", checked);
}

// ---- Output parsing ----

struct Want {
  Kind kind;
  std::string text;  // kContent / kReasoning
  std::string name;  // kToolCall
  Json arguments;    // kToolCall
};

Want Content(std::string t) {
  return {Kind::kContent, std::move(t), {}, {}};
}
Want Reasoning(std::string t) {
  return {Kind::kReasoning, std::move(t), {}, {}};
}
Want Call(std::string name, Json args) {
  return {Kind::kToolCall, {}, std::move(name), std::move(args)};
}

struct ParseCase {
  const char* name;
  bool thinking;
  bool tools;
  bool prefill;
  std::string output;
  std::vector<Want> want;
};

const Json& Tools() {
  static const Json tools = Json::parse(R"([
    {"type": "function", "function": {"name": "get_weather",
      "parameters": {"type": "object", "properties": {
        "city": {"type": "string"}, "days": {"type": "integer"},
        "metric": {"type": "boolean"}, "ratio": {"type": "number"},
        "opts": {"type": "object"}, "tags": {"type": "array"},
        "maybe": {"type": ["integer", "null"]},
        "either": {"anyOf": [{"type": "number"}, {"type": "string"}]}}}}}
  ])");
  return tools;
}

ChatPrompt PromptFor(const ParseCase& c) {
  ChatPrompt prompt;
  prompt.messages = Json::array({{{"role", "user"}, {"content", "x"}}});
  if (c.prefill) {
    prompt.messages.push_back({{"role", "assistant"}, {"content", "y"}});
  }
  prompt.thinking = c.thinking;
  if (c.tools) {
    prompt.tools = Tools();
  }
  return prompt;
}

// Adjacent text parts of one kind merged: how they reach a client.
std::vector<OutputPart> Merge(const std::vector<OutputPart>& parts) {
  std::vector<OutputPart> out;
  for (const auto& p : parts) {
    if (p.kind != Kind::kToolCall && !out.empty() &&
        out.back().kind == p.kind) {
      out.back().text += p.text;
    } else {
      out.push_back(p);
    }
  }
  return out;
}

std::vector<OutputPart> ParseSplit(const ParseCase& c,
                                   const std::vector<std::size_t>& cuts) {
  const auto format = NewQwen35ChatFormat();
  auto parser = format->NewParser(PromptFor(c));
  std::vector<OutputPart> parts;
  std::size_t start = 0;
  auto take = [&](std::vector<OutputPart> more) {
    parts.insert(parts.end(), more.begin(), more.end());
  };
  for (const std::size_t cut : cuts) {
    take(parser->Feed(c.output.substr(start, cut - start)));
    start = cut;
  }
  take(parser->Feed(c.output.substr(start)));
  take(parser->Finish());
  return Merge(parts);
}

std::string Describe(const std::vector<OutputPart>& parts) {
  std::string out;
  for (const auto& p : parts) {
    out += p.kind == Kind::kContent ? "[content \"" + Show(p.text) + "\"]"
           : p.kind == Kind::kReasoning
               ? "[reasoning \"" + Show(p.text) + "\"]"
               : "[call " + p.tool_name + " " + p.arguments + "]";
  }
  return out;
}

bool Matches(const std::vector<OutputPart>& got,
             const std::vector<Want>& want) {
  if (got.size() != want.size()) {
    return false;
  }
  for (std::size_t i = 0; i < got.size(); ++i) {
    if (got[i].kind != want[i].kind) {
      return false;
    }
    if (got[i].kind == Kind::kToolCall) {
      const auto args = Json::parse(got[i].arguments, nullptr, false);
      if (got[i].tool_name != want[i].name || args != want[i].arguments ||
          args.dump() != want[i].arguments.dump() ||
          got[i].tool_id.rfind("call_", 0) != 0 ||
          got[i].tool_id.size() != 29) {
        return false;
      }
    } else if (got[i].text != want[i].text) {
      return false;
    }
  }
  return true;
}

const std::vector<ParseCase>& ParseCases() {
  static const std::vector<ParseCase> cases = {
      {"plain",
       false,
       false,
       false,
       "Hello, world!\nLine two.",
       {Content("Hello, world!\nLine two.")}},
      {"edge_newlines",
       false,
       false,
       false,
       "\n\nHi\n\n\nthere\n\n",
       {Content("Hi\n\n\nthere")}},
      {"thinking",
       true,
       false,
       false,
       "Let me think.\nStep 2.\n</think>\n\nThe answer is 4.",
       {Reasoning("Let me think.\nStep 2."), Content("The answer is 4.")}},
      {"thinking_unterminated",
       true,
       false,
       false,
       "Still thinking\n",
       {Reasoning("Still thinking")}},
      {"thinking_repeated_open",
       true,
       false,
       false,
       "<think>\nreason\n</think>\n\nok",
       {Reasoning("reason"), Content("ok")}},
      {"thinking_starts_with_markup",
       true,
       false,
       false,
       "<b>bold</b> idea\n</think>\n\nok",
       {Reasoning("<b>bold</b> idea"), Content("ok")}},
      {"tool_call",
       false,
       true,
       false,
       "I'll check.\n\n<tool_call>\n<function=get_weather>\n"
       "<parameter=city>\nParis\n</parameter>\n<parameter=days>\n3\n"
       "</parameter>\n</function>\n</tool_call>",
       {Content("I'll check."),
        Call("get_weather", Json::parse(R"({"city": "Paris", "days": 3})"))}},
      {"typed_values",
       false,
       true,
       false,
       "<tool_call>\n<function=get_weather>\n"
       "<parameter=city>\n123\n</parameter>\n"
       "<parameter=days>\nthree\n</parameter>\n"
       "<parameter=metric>\nTrue\n</parameter>\n"
       "<parameter=ratio>\n2.5\n</parameter>\n"
       "<parameter=opts>\n{\"a\": [1, 2]}\n</parameter>\n"
       "<parameter=tags>\n[\"x\"]\n</parameter>\n"
       "<parameter=maybe>\nNone\n</parameter>\n"
       "<parameter=either>\n12.5\n</parameter>\n"
       "<parameter=extra>\n42\n</parameter>\n"
       "</function>\n</tool_call>",
       {Call("get_weather",
             Json::parse(R"({"city": "123", "days": "three", "metric": true,
                             "ratio": 2.5, "opts": {"a": [1, 2]}, "tags": ["x"],
                             "maybe": null, "either": 12.5, "extra": "42"})"))}},
      {"either_text",
       false,
       true,
       false,
       "<tool_call>\n<function=get_weather>\n<parameter=either>\nabc\n"
       "</parameter>\n<parameter=maybe>\n7\n</parameter>\n</function>\n"
       "</tool_call>",
       {Call("get_weather", Json::parse(R"({"either": "abc", "maybe": 7})"))}},
      {"two_calls_with_thinking",
       true,
       true,
       false,
       "Two cities.\n</think>\n\nI'll check both.\n\n<tool_call>\n"
       "<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n"
       "</function>\n</tool_call>\n<tool_call>\n<function=get_weather>\n"
       "<parameter=city>\nRome\n</parameter>\n</function>\n</tool_call>",
       {Reasoning("Two cities."), Content("I'll check both."),
        Call("get_weather", Json::parse(R"({"city": "Paris"})")),
        Call("get_weather", Json::parse(R"({"city": "Rome"})"))}},
      {"multiline_value",
       false,
       true,
       false,
       "<tool_call>\n<function=get_weather>\n<parameter=city>\nNew\nYork\n\n"
       "</parameter>\n</function>\n</tool_call>",
       {Call("get_weather", Json::parse(R"({"city": "New\nYork\n"})"))}},
      {"no_tools_markup_is_content",
       false,
       false,
       false,
       "<tool_call>\n<function=x>\n</function>\n</tool_call>",
       {Content("<tool_call>\n<function=x>\n</function>\n</tool_call>")}},
      {"malformed_call",
       false,
       true,
       false,
       "Text\n\n<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n"
       "</function>\n</tool_call> after",
       {Content("Text\n\n<tool_call>\n<function=get_weather>\n"
                "<parameter=city>\nParis\n</function>\n</tool_call> after")}},
      {"unterminated_call",
       false,
       true,
       false,
       "Sure.\n<tool_call>\n<function=get_weather>\n<parameter=city>\nPar",
       {Content("Sure.\n<tool_call>\n<function=get_weather>\n"
                "<parameter=city>\nPar")}},
      {"partial_markers_are_text",
       false,
       true,
       false,
       "a < b and <tool or <tool_cal",
       {Content("a < b and <tool or <tool_cal")}},
      {"content_after_call",
       false,
       true,
       false,
       "<tool_call>\n<function=get_weather>\n</function>\n</"
       "tool_call>\n\nDone.",
       {Call("get_weather", Json::object()), Content("Done.")}},
      {"unknown_function",
       false,
       true,
       false,
       "<tool_call>\n<function=other>\n<parameter=n>\n5\n</parameter>\n"
       "</function>\n</tool_call>",
       {Call("other", Json::parse(R"({"n": "5"})"))}},
      {"unicode",
       false,
       false,
       false,
       "Ünïcödé 🦊 <think> text",
       {Content("Ünïcödé 🦊 <think> text")}},
      {"prefill_is_content",
       true,
       false,
       true,
       "more text</think>",
       {Content("more text</think>")}},
  };
  return cases;
}

void CheckParse() {
  int runs = 0;
  for (const auto& c : ParseCases()) {
    const std::size_t n = c.output.size();
    std::vector<std::vector<std::size_t>> splits = {{}};
    std::vector<std::size_t> every;
    for (std::size_t i = 1; i < n; ++i) {
      splits.push_back({i});
      every.push_back(i);
      for (std::size_t j = i + 1; j < n && n <= 160; ++j) {
        splits.push_back({i, j});
      }
    }
    splits.push_back(every);  // one byte at a time
    for (const auto& cuts : splits) {
      const auto got = ParseSplit(c, cuts);
      ++runs;
      if (!Matches(got, c.want)) {
        std::string at;
        for (const auto k : cuts) {
          at += " " + std::to_string(k);
        }
        Fail(std::string("parse ") + c.name + " split at" +
             (cuts.size() > 3 ? std::string(" every byte") : at) + ": got " +
             Describe(got));
        break;
      }
    }
  }
  std::printf("ok: %zu parse cases over %d splits\n", ParseCases().size(),
              runs);
}

// Nothing is held back that cannot still become markup.
void CheckHoldBack() {
  const auto format = NewQwen35ChatFormat();
  ChatPrompt with_tools;
  with_tools.messages = Json::array({{{"role", "user"}, {"content", "x"}}});
  with_tools.tools = Tools();
  ChatPrompt plain = with_tools;
  plain.tools = Json::array();
  ChatPrompt thinking = plain;
  thinking.thinking = true;

  auto text = [](const std::vector<OutputPart>& parts) {
    std::string out;
    for (const auto& p : parts) {
      out += p.text;
    }
    return out;
  };
  auto p = format->NewParser(with_tools);
  if (text(p->Feed("Hello <tool")) != "Hello ") {
    Fail("hold back: a possible <tool_call> start");
  }
  const auto call =
      p->Feed("_call>\n<function=get_weather>\n</function>\n</tool_call>");
  if (call.size() != 1 || call[0].kind != Kind::kToolCall) {
    Fail("hold back: the call after a held start");
  }
  p = format->NewParser(plain);
  if (text(p->Feed("Hello <tool")) != "Hello <tool") {
    Fail("hold back: no tools, so <tool is text");
  }
  if (text(p->Feed("abc\n")) != "abc" || text(p->Feed("def")) != "\ndef") {
    Fail("hold back: a trailing newline until more text");
  }
  p = format->NewParser(thinking);
  if (text(p->Feed("think </th")) != "think " ||
      text(p->Feed("ink>answer")) != "answer") {
    Fail("hold back: a possible </think>");
  }
  std::printf("ok: hold-back\n");
}

// Parsing a reply and rendering it back as the assistant turn gives the
// prompt plus the reply: the next request's prompt starts with this one's.
void CheckRoundTrip() {
  const auto format = NewQwen35ChatFormat();
  for (const auto& c : ParseCases()) {
    const std::string name = c.name;
    if (name != "two_calls_with_thinking" && name != "tool_call" &&
        name != "thinking" && name != "plain") {
      continue;
    }
    ChatPrompt prompt = PromptFor(c);
    auto parser = format->NewParser(prompt);
    auto parts = parser->Feed(c.output);
    const auto tail = parser->Finish();
    parts.insert(parts.end(), tail.begin(), tail.end());
    Json assistant = {{"role", "assistant"}};
    std::string content;
    std::string reasoning;
    Json calls = Json::array();
    for (const auto& p : Merge(parts)) {
      if (p.kind == Kind::kContent) {
        content += p.text;
      } else if (p.kind == Kind::kReasoning) {
        reasoning += p.text;
      } else {
        calls.push_back(
            {{"id", p.tool_id},
             {"type", "function"},
             {"function",
              {{"name", p.tool_name}, {"arguments", p.arguments}}}});
      }
    }
    assistant["content"] = content;
    if (!reasoning.empty()) {
      assistant["reasoning_content"] = reasoning;
    }
    if (!calls.empty()) {
      assistant["tool_calls"] = calls;
    }
    ChatPrompt next = prompt;
    next.messages.push_back(assistant);
    const std::string want = format->Render(prompt) + c.output;
    if (format->Render(next) != want) {
      Fail("round trip " + name + ":\n  got  " + Show(format->Render(next)) +
           "\n  want " + Show(want));
    }
  }
  std::printf("ok: parse then render reproduces the reply\n");
}

// `gufo-npu generate` renders with qwen35::RenderChat (system and user
// messages only); the server's renderer must give the same prompt.
void CheckCliRenderChat() {
  namespace qwen35 = gufo_npu::qwen35;
  const auto format = NewQwen35ChatFormat();
  const std::vector<std::vector<qwen35::Message>> cases = {
      {{"user", "Why is the sky blue?"}},
      {{"system", "  Be brief. "}, {"user", "\nHello\n"}},
      {{"user", "Ünïcödé 🦊"}},
  };
  for (const auto& messages : cases) {
    for (const bool thinking : {false, true}) {
      ChatPrompt prompt;
      prompt.thinking = thinking;
      for (const auto& m : messages) {
        prompt.messages.push_back({{"role", m.role}, {"content", m.content}});
      }
      if (format->Render(prompt) != qwen35::RenderChat(messages, thinking)) {
        Fail("qwen35::RenderChat differs from the server's render for " +
             Show(messages.back().content));
      }
    }
  }
  std::printf("ok: qwen35::RenderChat matches\n");
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 2) {
    std::fprintf(stderr, "usage: chat_format_test FIXTURE_DIR\n");
    return 2;
  }
  try {
    CheckRender(argv[1]);
    CheckParse();
    CheckHoldBack();
    CheckRoundTrip();
    CheckCliRenderChat();
    const auto stops = NewQwen35ChatFormat()->StopTokens();
    if (stops != std::vector<std::string>{"<|im_end|>", "<|endoftext|>"}) {
      Fail("stop tokens");
    }
  } catch (const std::exception& e) {
    Fail(std::string("exception: ") + e.what());
  }
  return failures == 0 ? 0 : 1;
}
