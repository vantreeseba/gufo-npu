// The server end to end with the real Qwen3.5 chat format over a scripted
// engine (tests/serve/fakes.hpp): reasoning and tool calls come back as
// OpenAI message parts, streaming and not, and a tool result round trip
// renders the prompt the chat format renders for the same messages. Image
// parts (data URIs of tests/fixtures/image): refused by a model without
// image support, expanded to one pad per image token with the decoded
// image and its span for one that has it, and the requests refused.
#include <httplib/httplib.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <memory>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "fakes.hpp"
#include "serve/chat_format.hpp"
#include "serve/openai.hpp"
#include "serve/server.hpp"

namespace {

using gufo_npu::serve::ChatFormat;
using gufo_npu::serve::ChatPrompt;
using gufo_npu::serve::ImageLayout;
using gufo_npu::serve::Server;
using gufo_npu::serve::ServerOptions;
using gufo_npu::serve::testing::FakeEngine;
using nlohmann::json;
using nlohmann::ordered_json;

int failures = 0;

void Check(bool ok, const std::string& what) {
  std::printf("%s %s\n", ok ? "ok  " : "FAIL", what.c_str());
  if (!ok) {
    ++failures;
  }
}

struct Running {
  Running(FakeEngine& engine, const ChatFormat& format)
      : server(engine, format, [] {
          ServerOptions options;
          options.host = "127.0.0.1";
          options.port = 0;
          return options;
        }()) {
    port = server.Bind();
    thread = std::thread([this] { server.Run(); });
  }
  ~Running() {
    server.Stop();
    thread.join();
  }
  httplib::Client Client() const {
    httplib::Client c("127.0.0.1", port);
    c.set_read_timeout(30, 0);
    return c;
  }

  Server server;
  int port = 0;
  std::thread thread;
};

json Post(const Running& s, const ordered_json& body) {
  auto res =
      s.Client().Post("/v1/chat/completions", body.dump(), "application/json");
  if (!res || res->status != 200) {
    std::fprintf(stderr, "status %d: %s\n", res ? res->status : -1,
                 res ? res->body.c_str() : "");
    return json();
  }
  return json::parse(res->body, nullptr, false);
}

// The `data:` events of a streamed request, without the final [DONE].
std::vector<json> Stream(const Running& s, ordered_json body) {
  body["stream"] = true;
  auto res =
      s.Client().Post("/v1/chat/completions", body.dump(), "application/json");
  std::vector<json> events;
  if (!res || res->status != 200) {
    return events;
  }
  std::size_t at = 0;
  const std::string& text = res->body;
  while ((at = text.find("data: ", at)) != std::string::npos) {
    const auto end = text.find("\n\n", at);
    const auto data = text.substr(at + 6, end - at - 6);
    at = end;
    if (data != "[DONE]") {
      events.push_back(json::parse(data, nullptr, false));
    }
  }
  return events;
}

ordered_json WeatherTool() {
  return ordered_json::parse(R"({
    "type": "function",
    "function": {
      "name": "get_weather",
      "description": "Weather for a city",
      "parameters": {
        "type": "object",
        "properties": {
          "city": {"type": "string"},
          "days": {"type": "integer"}
        },
        "required": ["city"]
      }
    }
  })");
}

// A tool call in Qwen3.5's markup, split across tokens at awkward places.
std::vector<std::string> ToolCallPieces() {
  return {"I'll check.\n\n<tool", "_call>\n<function=get_",
          "weather>\n<parameter=city>\nPar",
          "is\n</parameter>\n<parameter=days>\n3\n</para",
          "meter>\n</function>\n</tool_call>"};
}

void TestReasoning(FakeEngine& engine, const Running& s) {
  engine.Script({"The sky", " scatters blue light.", "\n</think>\n\n",
                 "Rayleigh ", "scattering."});
  ordered_json body = {
      {"messages", {{{"role", "user"}, {"content", "Why is the sky blue?"}}}},
      {"chat_template_kwargs", {{"enable_thinking", true}}}};
  const auto r = Post(s, body);
  const auto& message = r["choices"][0]["message"];
  Check(
      message.value("reasoning_content", "") == "The sky scatters blue light.",
      "reasoning: reasoning_content");
  Check(message.value("content", "") == "Rayleigh scattering.",
        "reasoning: content");

  std::string reasoning;
  std::string content;
  for (const auto& e : Stream(s, body)) {
    const auto& delta = e["choices"][0]["delta"];
    if (delta.contains("reasoning_content")) {
      reasoning += delta["reasoning_content"].get<std::string>();
    }
    if (delta.contains("content") && delta["content"].is_string()) {
      content += delta["content"].get<std::string>();
    }
  }
  Check(reasoning == "The sky scatters blue light." &&
            content == "Rayleigh scattering.",
        "reasoning: streamed deltas");
}

void TestToolRoundTrip(FakeEngine& engine, const Running& s,
                       const ChatFormat& format) {
  ordered_json messages = {
      {{"role", "user"}, {"content", "Weather in Paris for 3 days?"}}};
  ordered_json tools = ordered_json::array({WeatherTool()});

  engine.Script(ToolCallPieces());
  const auto r = Post(s, {{"messages", messages}, {"tools", tools}});
  const auto& choice = r["choices"][0];
  const auto& calls = choice["message"]["tool_calls"];
  Check(choice.value("finish_reason", "") == "tool_calls",
        "tool call: finish_reason tool_calls");
  Check(calls.is_array() && calls.size() == 1 &&
            calls[0]["function"]["name"] == "get_weather",
        "tool call: one get_weather call");
  const auto args =
      calls.is_array() && !calls.empty()
          ? json::parse(calls[0]["function"]["arguments"].get<std::string>(),
                        nullptr, false)
          : json();
  Check(args == json({{"city", "Paris"}, {"days", 3}}),
        "tool call: arguments typed by the schema");
  Check(choice["message"].value("content", "") == "I'll check.",
        "tool call: the text before it is content");

  // Streaming: the same call arrives as tool_calls deltas.
  std::string name;
  std::string arguments;
  std::string finish;
  for (const auto& e : Stream(s, {{"messages", messages}, {"tools", tools}})) {
    const auto& c = e["choices"];
    if (!c.is_array() || c.empty()) {
      continue;
    }
    const auto& delta = c[0]["delta"];
    if (delta.contains("tool_calls")) {
      const auto& f = delta["tool_calls"][0]["function"];
      name += f.value("name", "");
      arguments += f.value("arguments", "");
    }
    if (c[0]["finish_reason"].is_string()) {
      finish = c[0]["finish_reason"];
    }
  }
  Check(name == "get_weather" &&
            json::parse(arguments, nullptr, false) == args &&
            finish == "tool_calls",
        "tool call: streamed deltas");

  // The client runs the tool and sends the result back.
  ordered_json assistant = {{"role", "assistant"},
                            {"content", "I'll check."},
                            {"tool_calls", ordered_json::array()}};
  assistant["tool_calls"].push_back(
      {{"id", calls.is_array() && !calls.empty() ? calls[0]["id"] : "call_x"},
       {"type", "function"},
       {"function",
        {{"name", "get_weather"},
         {"arguments", R"({"city": "Paris", "days": 3})"}}}});
  messages.push_back(assistant);
  messages.push_back({{"role", "tool"},
                      {"tool_call_id", assistant["tool_calls"][0]["id"]},
                      {"content", "Sunny, 21 C."}});
  engine.Script({"Sunny", " and 21 C."});
  const auto final = Post(s, {{"messages", messages}, {"tools", tools}});
  Check(final["choices"][0]["message"].value("content", "") ==
                "Sunny and 21 C." &&
            final["choices"][0].value("finish_reason", "") == "stop",
        "tool result: final answer");

  ChatPrompt prompt;
  prompt.messages = messages;
  prompt.tools = tools;
  const std::string rendered = format.Render(prompt);
  Check(engine.last().prompt.size() == rendered.size() &&
            rendered.find("<tool_response>\nSunny, 21 C.\n</tool_response>") !=
                std::string::npos,
        "tool result: the engine got the chat format's prompt");
}

// ---- Images ----

constexpr int kImagePad = 248056;

// A chat model that takes images: one id per byte of text (the byte's
// value), <|image_pad|> as Qwen's id, and a layout that rounds each side
// up to 32 pixels (16-pixel patches, 2 x 2 merged).
class ImageEngine : public FakeEngine {
public:
  ImageEngine(bool chat_images, bool takes_mmproj) {
    info_.chat_images = chat_images;
    info_.takes_mmproj = takes_mmproj;
    info_.vocab = 248320;
  }
  std::vector<int> Encode(std::string_view text) const override {
    std::vector<int> ids;
    for (const unsigned char c : text) {
      ids.push_back(c);
    }
    return ids;
  }
  int FindToken(std::string_view token) const override {
    return token == "<|image_pad|>" ? kImagePad : FakeEngine::FindToken(token);
  }
  ImageLayout ChatImageLayout(int width, int height) const override {
    if (std::max(width, height) > 200 * std::min(width, height)) {
      throw std::invalid_argument("aspect ratio above 200");
    }
    ImageLayout layout;
    layout.width = (width + 31) / 32 * 32;
    layout.height = (height + 31) / 32 * 32;
    layout.grid_w = layout.width / 16;
    layout.grid_h = layout.height / 16;
    layout.tokens = layout.grid_w * layout.grid_h / 4;
    return layout;
  }
};

std::vector<std::uint8_t> ReadAll(const std::filesystem::path& path) {
  std::ifstream in(path, std::ios::binary);
  return {std::istreambuf_iterator<char>(in), {}};
}

std::string ToBase64(const std::vector<std::uint8_t>& bytes) {
  static const char kAlphabet[] =
      "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
  std::string out;
  for (std::size_t i = 0; i < bytes.size(); i += 3) {
    const unsigned n = bytes[i] << 16 |
                       (i + 1 < bytes.size() ? bytes[i + 1] << 8 : 0) |
                       (i + 2 < bytes.size() ? bytes[i + 2] : 0);
    out += kAlphabet[n >> 18 & 63];
    out += kAlphabet[n >> 12 & 63];
    out += i + 1 < bytes.size() ? kAlphabet[n >> 6 & 63] : '=';
    out += i + 2 < bytes.size() ? kAlphabet[n & 63] : '=';
  }
  return out;
}

std::string DataUri(const std::filesystem::path& file, const char* type) {
  return std::string("data:") + type + ";base64," + ToBase64(ReadAll(file));
}

ordered_json ImagePart(const std::string& url) {
  return {{"type", "image_url"}, {"image_url", {{"url", url}}}};
}

ordered_json TextPart(const std::string& text) {
  return {{"type", "text"}, {"text", text}};
}

ordered_json UserParts(ordered_json parts) {
  return {{"messages",
           ordered_json::array({{{"role", "user"}, {"content", parts}}})}};
}

httplib::Result PostRaw(const Running& s, const ordered_json& body) {
  return s.Client().Post("/v1/chat/completions", body.dump(),
                         "application/json");
}

// "param/code" of an error reply.
std::string ErrorParam(const httplib::Result& r) {
  if (!r) {
    return "no response";
  }
  const auto body = json::parse(r->body, nullptr, false);
  if (!body.is_object() || !body.contains("error")) {
    return "no error body";
  }
  const auto& e = body["error"];
  return (e["param"].is_string() ? e["param"].get<std::string>() : "") + "/" +
         (e["code"].is_string() ? e["code"].get<std::string>() : "");
}

std::vector<int> Bytes(const std::string& text) {
  std::vector<int> ids;
  for (const unsigned char c : text) {
    ids.push_back(c);
  }
  return ids;
}

void TestImagesRefusedWithoutSupport(const ChatFormat& format,
                                     const std::filesystem::path& fixtures) {
  const auto body =
      UserParts({TextPart("What is this?"),
                 ImagePart(DataUri(fixtures / "rgb8.png", "image/png"))});
  {
    ImageEngine engine(false, false);  // every chat model today
    Running s(engine, format);
    const auto r = PostRaw(s, body);
    Check(r && r->status == 400 && ErrorParam(r) == "messages/" &&
              r->body.find("takes no images") != std::string::npos &&
              r->body.find("--mmproj") != std::string::npos,
          "images: a model without image support gives 400 for messages");
    Check(engine.generations() == 0, "images: refused before generating");
  }
  {
    ImageEngine engine(false, true);  // takes a vision tower, started without
    Running s(engine, format);
    const auto r = PostRaw(s, body);
    Check(
        r && r->status == 400 && ErrorParam(r) == "messages/" &&
            r->body.find("start the server with --mmproj") != std::string::npos,
        "images: a model started without --mmproj names it");
  }
}

void TestImagePrompt(const ChatFormat& format,
                     const std::filesystem::path& fixtures) {
  ImageEngine engine(true, true);
  Running s(engine, format);
  engine.Script({"A", " cat."});
  const std::string png = DataUri(fixtures / "rgb8.png", "image/png");
  const std::string jpeg = DataUri(fixtures / "jpeg420.jpg", "image/jpeg");

  // One image between text: 37 x 29 pixels -> 64 x 32, a 4 x 2 patch grid,
  // 2 tokens.
  auto r = Post(s, UserParts({TextPart("Describe "), ImagePart(png),
                              TextPart(" please")}));
  auto gen = engine.last();
  std::vector<int> want = Bytes("<|im_start|>user\nDescribe <|vision_start|>");
  const auto offset = static_cast<int>(want.size());
  want.insert(want.end(), 2, kImagePad);
  const auto tail = Bytes(
      "<|vision_end|> please<|im_end|>\n<|im_start|>assistant\n"
      "<think>\n\n</think>\n\n");
  want.insert(want.end(), tail.begin(), tail.end());
  Check(r["choices"][0]["message"].value("content", "") == "A cat.",
        "image: the reply");
  Check(gen.prompt == want, "image: text ids, the pads, text ids");
  Check(gen.images.size() == 1 && gen.images[0].offset == offset &&
            gen.images[0].layout.tokens == 2 &&
            gen.images[0].layout.grid_w == 4 &&
            gen.images[0].layout.grid_h == 2 &&
            gen.images[0].layout.width == 64 &&
            gen.images[0].layout.height == 32,
        "image: its span and layout");
  Check(!gen.images.empty() && gen.images[0].image->width == 37 &&
            gen.images[0].image->height == 29 &&
            gen.images[0].image->rgb == ReadAll(fixtures / "rgb8.rgb"),
        "image: decoded to Pillow's pixels");
  Check(r["usage"]["prompt_tokens"] == static_cast<int>(want.size()),
        "image: usage counts the image tokens");
  bool checked = true;
  try {
    gufo_npu::serve::CheckPromptImages(gen, kImagePad);
  } catch (const std::invalid_argument&) {
    checked = false;
  }
  Check(checked, "image: CheckPromptImages accepts the server's prompt");

  // Two images in order, over two turns, streamed; `detail` is accepted.
  ordered_json jpeg_part = ImagePart(jpeg);
  jpeg_part["image_url"]["detail"] = "low";
  ordered_json body = {
      {"messages",
       ordered_json::array(
           {{{"role", "user"},
             {"content", ordered_json::array({ImagePart(png), TextPart("?")})}},
            {{"role", "assistant"}, {"content", "A cat."}},
            {{"role", "user"},
             {"content", ordered_json::array({jpeg_part})}}})}};
  const auto events = Stream(s, body);
  gen = engine.last();
  Check(!events.empty() && gen.images.size() == 2 &&
            gen.images[0].offset < gen.images[1].offset &&
            gen.images[1].image->rgb == ReadAll(fixtures / "jpeg420.rgb") &&
            gen.images[0].hash != gen.images[1].hash,
        "two images: in order, each its own");
  checked = true;
  try {
    gufo_npu::serve::CheckPromptImages(gen, kImagePad);
  } catch (const std::invalid_argument&) {
    checked = false;
  }
  Check(checked, "two images: CheckPromptImages accepts them");

  // What is refused: 400 for messages.
  const auto refused = [&](const ordered_json& b, const std::string& what) {
    const auto res = PostRaw(s, b);
    Check(res && res->status == 400 && ErrorParam(res) == "messages/",
          what + ": 400 for messages");
  };
  const auto one = [&](const ordered_json& part) {
    return UserParts({TextPart("x"), part});
  };
  refused(one(ImagePart("https://example.com/a.png")), "an image URL");
  refused(one(ImagePart("/tmp/a.png")), "an image path");
  refused(one(ImagePart("data:image/png;base64,@@@@")), "invalid base64");
  refused(one(ImagePart("data:image/png;base64," +
                        ToBase64({'h', 'e', 'l', 'l', 'o'}))),
          "bytes that are no image");
  refused(one(ImagePart(DataUri(fixtures / "truncated.png", "image/png"))),
          "a truncated PNG");
  refused(one(ImagePart(DataUri(fixtures / "huge.png", "image/png"))),
          "an image over the pixel limit");
  refused(one({{"type", "image_url"}, {"image_url", png}}),
          "image_url as a string");
  refused(one({{"type", "image"}, {"image", png}}), "a type image part");
  refused(one({{"type", "input_image"}, {"image_url", png}}),
          "an input_image part");
  ordered_json bad_detail = ImagePart(png);
  bad_detail["image_url"]["detail"] = "ultra";
  refused(one(bad_detail), "an unknown detail");
  ordered_json extra = ImagePart(png);
  extra["image_url"]["size"] = 3;
  refused(one(extra), "an unknown image_url field");
  ordered_json many = ordered_json::array();
  for (int i = 0; i <= gufo_npu::serve::kMaxChatImages; ++i) {
    many.push_back(ImagePart(png));
  }
  refused(UserParts(many), "too many images");
  refused(
      {{"messages", ordered_json::array(
                        {{{"role", "system"},
                          {"content", ordered_json::array({ImagePart(png)})}},
                         {{"role", "user"}, {"content", "hi"}}})}},
      "an image in the system message");
  refused(UserParts({TextPart("a <|image_pad|> b"), ImagePart(png)}),
          "an image pad in the text");
  const int before = engine.generations();
  refused(UserParts({TextPart("plain <|image_pad|>")}),
          "an image pad in text without images");
  Check(engine.generations() == before, "refusals generate nothing");
}

void TestCheckPromptImages() {
  using gufo_npu::serve::CheckPromptImages;
  using gufo_npu::serve::GenerateRequest;
  using gufo_npu::serve::PromptImage;
  const auto image = std::make_shared<const gufo_npu::image::Decoded>();
  const auto request = [&](std::vector<int> prompt,
                           std::vector<std::pair<int, int>> spans) {
    GenerateRequest r;
    r.prompt = std::move(prompt);
    for (const auto& [offset, tokens] : spans) {
      PromptImage img;
      img.image = image;
      img.offset = offset;
      img.layout.tokens = tokens;
      r.images.push_back(img);
    }
    return r;
  };
  const auto ok = [&](const GenerateRequest& r) {
    try {
      CheckPromptImages(r, kImagePad);
      return true;
    } catch (const std::invalid_argument&) {
      return false;
    }
  };
  const int p = kImagePad;
  Check(ok(request({1, p, p, 2, p, 3}, {{1, 2}, {4, 1}})),
        "CheckPromptImages: two runs");
  Check(ok(request({1, 2}, {})), "CheckPromptImages: no images");
  Check(!ok(request({1, p, p, 2}, {{1, 1}})),
        "CheckPromptImages: a pad outside the runs");
  Check(!ok(request({1, p, 2}, {{1, 2}})),
        "CheckPromptImages: a run past its pads");
  Check(!ok(request({1, p, p}, {{1, 3}})),
        "CheckPromptImages: a run past the prompt");
  Check(!ok(request({p, p, 1, p}, {{3, 1}, {0, 2}})),
        "CheckPromptImages: runs out of order");
  Check(!ok(request({1, p}, {{1, 0}})), "CheckPromptImages: an empty run");
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 2) {
    std::fprintf(stderr, "usage: serve_chat_test <image-fixtures>\n");
    return 2;
  }
  const auto format = gufo_npu::serve::NewQwen35ChatFormat();
  {
    FakeEngine engine;
    Running s(engine, *format);
    TestReasoning(engine, s);
    TestToolRoundTrip(engine, s, *format);
  }
  TestImagesRefusedWithoutSupport(*format, argv[1]);
  TestImagePrompt(*format, argv[1]);
  TestCheckPromptImages();
  return failures == 0 ? 0 : 1;
}
