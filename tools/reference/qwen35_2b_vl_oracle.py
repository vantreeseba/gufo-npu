#!/usr/bin/env python3
"""HF transformers oracle for Qwen3.5-2B image prompts (vision tower, MRoPE,
image rows in the text model), fp32 on the CPU.

  python -I qwen35_2b_vl_oracle.py --model /models/Qwen3.5-2B-hf \\
      --out /models/oracle-2b-vl

The inputs are synthetic (cards and a text page drawn with Pillow), so the
oracle needs no downloads; the exact PNGs are written next to the results.
Each case is a single user message rendered with the model's chat template
(add_generation_prompt, thinking off: the template's default, as in
qwen35_oracle.py), run through Qwen3VLProcessor with the image-token cap
gufo-npu serves with (smart_resize factor 32, 65,536 to 1,843,200 pixels,
i.e. 64 to 1,800 image tokens per image). The model generates a greedy
continuation, then runs once teacher-forced over prompt + continuation.

Per case, in --out:
  <case>.imgN.png       the input images, as given to the processor (RGB8)
  <case>.ids.json       {"ids": prompt + continuation, image pads expanded,
                         "prompt_len": n}
  <case>.positions.i32  [3, L] int32: get_rope_index rows t, h, w
  <case>.grid.json      [[t, h, w], ...] per image (patch grid)
  <case>.pixels.f32     [patches, 1536] the processor's pixel_values
  <case>.features.f32   [image tokens, 2048] the vision tower's merged output,
                        all images of the case in order
  <case>.logits.f32     [L - prompt_len + 1, vocab]: rows prompt_len - 1 .. L - 1
  <case>.hidden.f32     [layers + 1, L, 2048] (embeddings first), as
                        qwen35_oracle.py writes it
  <case>.greedy.json    the continuation's token ids
plus manifest.json (per case: prompt, content, images with their resize,
grid and token count, rope_delta, lengths, file shapes; library versions).

For each resized image the manifest also compares the processor's resize
(torchvision, uint8 antialiased bicubic) with Pillow's Image.resize bicubic,
and the vision tower's features from both. --resize-only runs just that
comparison (no model weights read) over the cases' images and a few
scan-sized pages, and prints it.
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import transformers
from PIL import Image, ImageDraw, ImageFont
import PIL

PATCH = 16
MERGE = 2
FACTOR = PATCH * MERGE
MIN_PIXELS = 64 * FACTOR * FACTOR  # preprocessor_config shortest_edge 65536
MAX_PIXELS = 1800 * FACTOR * FACTOR  # gufo-npu's --max-image-tokens 1800
SIZE = {"shortest_edge": MIN_PIXELS, "longest_edge": MAX_PIXELS}
IMAGE_PAD = 248056
VISION_START = 248053
VISION_END = 248054


def smart_resize(height: int, width: int) -> tuple[int, int]:
    """transformers' qwen2_vl smart_resize with the bounds above."""
    if max(height, width) / min(height, width) > 200:
        raise ValueError("aspect ratio above 200")
    h = round(height / FACTOR) * FACTOR
    w = round(width / FACTOR) * FACTOR
    if h * w > MAX_PIXELS:
        beta = math.sqrt((height * width) / MAX_PIXELS)
        h = max(FACTOR, math.floor(height / beta / FACTOR) * FACTOR)
        w = max(FACTOR, math.floor(width / beta / FACTOR) * FACTOR)
    elif h * w < MIN_PIXELS:
        beta = math.sqrt(MIN_PIXELS / (height * width))
        h = math.ceil(height * beta / FACTOR) * FACTOR
        w = math.ceil(width * beta / FACTOR) * FACTOR
    return h, w


# --- synthetic images -------------------------------------------------------

def font(size: int):
    # Pillow's bundled TrueType font (needs FreeType), so no system fonts.
    return ImageFont.load_default(size=size)


def shapes_card(width: int, height: int, seed: int) -> Image.Image:
    img = Image.new("RGB", (width, height), (240, 236, 220))
    d = ImageDraw.Draw(img)
    s = min(width, height)
    d.rectangle([s // 10, s // 10, s // 10 + s // 3, s // 10 + s // 4],
                fill=(200, 30, 40) if seed % 2 == 0 else (30, 60, 200))
    cx, cy, r = width * 2 // 3, height // 3, s // 6
    d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(30, 140, 60))
    d.polygon([(width // 4, height - s // 10), (width // 2, height // 2),
               (width * 3 // 4, height - s // 10)],
              fill=(240, 180, 20) if seed % 2 == 0 else (120, 40, 160))
    d.line([(0, height - 1), (width - 1, 0)], fill=(20, 20, 20), width=2)
    return img


def text_card(width: int, height: int, lines: list[str], size: int) -> Image.Image:
    img = Image.new("RGB", (width, height), (255, 255, 255))
    d = ImageDraw.Draw(img)
    f = font(size)
    y = size // 2
    for line in lines:
        d.text((size // 2, y), line, fill=(10, 10, 10), font=f)
        y += size * 5 // 4
    d.rectangle([0, 0, width - 1, height - 1], outline=(80, 80, 80), width=2)
    return img


def gradient_card(width: int, height: int, seed: int) -> Image.Image:
    """qwen3vl_embedding_oracle.py's test card: gradients, checks, dots."""
    px = bytearray()
    for y in range(height):
        for x in range(width):
            r = x * 255 // max(width - 1, 1)
            g = y * 255 // max(height - 1, 1)
            b = 30 + 200 * ((x // 8 + y // 8 + seed) % 2)
            if (x * 7 + y * 3 + seed) % 23 == 0:
                r, g, b = 255 - r, 255 - g, 255 - b
            px += bytes((r, g, b))
    return Image.frombytes("RGB", (width, height), bytes(px))


PAGE_TEXT = [
    ("h", "Harbour Street Allotment Society"),
    ("h", "Minutes of the spring meeting, 14 March 2026"),
    ("p", "Present: eleven members, the treasurer and the secretary. The chair "
          "opened the meeting at seven in the evening and thanked everyone for "
          "coming out in the rain."),
    ("p", "1. Water. The north tap leaked through February and was repaired on "
          "the 3rd for 48.50 EUR. Members are asked to close the valve after use "
          "and to report drips to the secretary."),
    ("p", "2. Plots. Plot 12 is free again; three people are on the waiting "
          "list. Plot 7 will be split into two half plots, 7a and 7b, from April. "
          "Paths between plots must stay at least 60 cm wide."),
    ("p", "3. Money. Fees for the year stay at 35 EUR per plot. The account "
          "holds 1,214.30 EUR after the tap repair and the new padlock (12.90 EUR)."),
    ("p", "4. Compost. The second bin is full. From now on only garden waste goes "
          "into the bins: no cooked food, no plastic pots, no weeds with seeds."),
    ("p", "5. Open day. Saturday 23 May, ten to four. Volunteers for the tea "
          "table and the seedling swap please sign the list on the shed door."),
    ("p", "Any other business: the hedge by the gate needs cutting before birds "
          "nest. The meeting closed at 20:40. Next meeting: 11 June 2026."),
]


def page(width: int, height: int) -> Image.Image:
    img = Image.new("RGB", (width, height), (252, 251, 247))
    d = ImageDraw.Draw(img)
    margin = 96
    body, head = font(30), font(40)
    y = margin
    for kind, text in PAGE_TEXT:
        f = head if kind == "h" else body
        words, line = text.split(), ""
        for word in words:
            trial = f"{line} {word}".strip()
            if d.textlength(trial, font=f) > width - 2 * margin and line:
                d.text((margin, y), line, fill=(20, 20, 30), font=f)
                y += int(f.size * 1.45)
                line = word
            else:
                line = trial
        d.text((margin, y), line, fill=(20, 20, 30), font=f)
        y += int(f.size * (1.6 if kind == "h" else 2.0))
    if y > height - margin:
        raise SystemExit(f"page text overflows ({y} > {height - margin})")
    d.text((width // 2 - 20, height - margin // 2 - 15), "- 1 -", fill=(90, 90, 90),
           font=font(24))
    return img


# (name, content): content items are ("image", PIL image) or ("text", str).
def cases() -> list[tuple[str, list]]:
    return [
        # 256 x 192 is below 65,536 pixels: upscaled to 320 x 224 (70 tokens).
        ("card_shapes", [("image", shapes_card(256, 192, 0)),
                         ("text", "Describe this image in one sentence.")]),
        # 333 x 251 rounds to 320 x 256: shrunk (80 tokens).
        ("card_text", [("image", text_card(333, 251, [
            "INVOICE 2025-0412", "Corner Hardware", "2 x screws, 1 x hammer",
            "Total: 23.40 EUR"], 28)),
                       ("text", "What does the card say?")]),
        # 320 x 256 is kept as is (80 tokens); text before and after the image.
        ("card_kept", [("text", "Look at this test card."),
                       ("image", gradient_card(320, 256, 1)),
                       ("text", "Which colours do you see, and where?")]),
        # Two images, both just under 65,536 pixels: 288 x 224 -> 320 x 256 (80
        # tokens) and 400 x 160 -> 416 x 192 (78 tokens).
        ("two_images", [("image", shapes_card(288, 224, 1)),
                        ("image", text_card(400, 160, ["OPEN 9-17", "Mon to Fri"], 40)),
                        ("text", "What is in the first picture, and what does "
                                 "the second one say?")]),
        # A page at the token cap: 1280 x 1440 = 1,843,200 pixels, kept (1,800 tokens).
        ("page", [("image", page(1280, 1440)),
                  ("text", "Transcribe the text on this page.")]),
    ]


SCAN_SIZES = [  # extra --resize-only inputs: (name, page or card, width, height)
    ("letter_150dpi", "page", 1275, 1650),
    ("a4_200dpi", "page", 1654, 2339),
    ("a4_300dpi", "page", 2480, 3508),
    ("photo_4x3", "card", 1000, 750),
    ("small", "card", 100, 70),
]


# --- resize comparison ------------------------------------------------------

def unpatchify(pixels: np.ndarray, grid: list[int]) -> np.ndarray:
    """pixel_values of one image -> (3, H, W) float, temporal slot 0."""
    _, gh, gw = grid
    p = pixels.reshape(gh // MERGE, gw // MERGE, MERGE, MERGE, 3, 2, PATCH, PATCH)
    p = p[:, :, :, :, :, 0]  # (bh, bw, mr, mc, c, y, x)
    return p.transpose(4, 0, 2, 5, 1, 3, 6).reshape(3, gh * PATCH, gw * PATCH)


def hf_resized_rgb(processor, image: Image.Image) -> tuple[np.ndarray, list[int], float]:
    """The processor's resized image as uint8 (3, H, W), its grid, and the
    largest distance of the de-normalized values from an integer."""
    out = processor.image_processor(images=[image], size=SIZE, return_tensors="pt")
    grid = out["image_grid_thw"][0].tolist()
    x = unpatchify(out["pixel_values"].numpy(), grid).astype(np.float64)
    x = (x * 0.5 + 0.5) * 255.0
    frac = float(np.abs(x - np.round(x)).max())
    return np.clip(np.round(x), 0, 255).astype(np.uint8), grid, frac


def pillow_resized_rgb(image: Image.Image, h: int, w: int) -> np.ndarray:
    if image.size == (w, h):
        r = image
    else:
        r = image.resize((w, h), Image.Resampling.BICUBIC)
    return np.asarray(r.convert("RGB")).transpose(2, 0, 1)


def torch_float_resized_rgb(image: Image.Image, h: int, w: int) -> np.ndarray:
    """Antialiased bicubic in float32 (torch.nn.functional.interpolate), rounded."""
    t = torch.from_numpy(np.asarray(image.convert("RGB")).copy()).permute(2, 0, 1)
    t = t[None].float()
    if (t.shape[2], t.shape[3]) != (h, w):
        t = F.interpolate(t, size=(h, w), mode="bicubic", align_corners=False,
                          antialias=True)
    return t.clamp(0, 255).round()[0].to(torch.uint8).numpy()


def diff_stats(a: np.ndarray, b: np.ndarray) -> dict:
    d = np.abs(a.astype(np.int32) - b.astype(np.int32))
    return {
        "max_abs": int(d.max()),
        "mean_abs": float(d.mean()),
        "values_differing": int((d != 0).sum()),
        "values": int(d.size),
    }


def resize_report(processor, image: Image.Image) -> dict:
    w0, h0 = image.size
    h, w = smart_resize(h0, w0)
    hf, grid, frac = hf_resized_rgb(processor, image)
    if grid != [1, h // PATCH, w // PATCH]:
        raise SystemExit(f"processor grid {grid} != smart_resize {h} x {w}")
    report = {
        "width": w0,
        "height": h0,
        "resized_width": w,
        "resized_height": h,
        "resized": (h, w) != (h0, w0),
        "grid": grid,
        "tokens": grid[1] * grid[2] // (MERGE * MERGE),
        "hf_integer_distance": frac,
    }
    if report["resized"]:
        report["hf_vs_pillow"] = diff_stats(hf, pillow_resized_rgb(image, h, w))
        report["hf_vs_torch_float"] = diff_stats(hf, torch_float_resized_rgb(image, h, w))
    return report


# --- the model run ----------------------------------------------------------

def render_expected(content: list) -> str:
    parts = []
    for kind, value in content:
        parts.append("<|vision_start|><|image_pad|><|vision_end|>" if kind == "image" else value)
    return ("<|im_start|>user\n" + "".join(parts) + "<|im_end|>\n"
            "<|im_start|>assistant\n<think>\n\n</think>\n\n")


def messages_for(content: list) -> list[dict]:
    items = []
    for kind, value in content:
        items.append({"type": "image", "image": value} if kind == "image"
                     else {"type": "text", "text": value})
    return [{"role": "user", "content": items}]


def write(path: Path, array: np.ndarray, dtype) -> dict:
    a = np.ascontiguousarray(array, dtype=dtype)
    a.tofile(path)
    return {"dtype": np.dtype(dtype).name, "shape": list(a.shape)}


def run_case(name, content, model, processor, tokenizer, out: Path, max_new: int) -> dict:
    images = [v for k, v in content if k == "image"]
    for i, img in enumerate(images):
        img.save(out / f"{name}.img{i}.png")
    messages = messages_for(content)
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True,
                                           tokenize=False)
    if prompt != render_expected(content):
        raise SystemExit(f"{name}: the chat template renders {prompt!r}")
    inputs = processor(text=[prompt], images=images, size=SIZE, return_tensors="pt")
    prompt_ids = inputs["input_ids"][0].tolist()
    grid = inputs["image_grid_thw"]
    mm = inputs["mm_token_type_ids"]
    if not torch.equal(mm[0].long(), (inputs["input_ids"][0] == IMAGE_PAD).long()):
        raise SystemExit(f"{name}: mm_token_type_ids is not the image-pad mask")
    reports = [resize_report(processor, img) for img in images]
    for r, g in zip(reports, grid.tolist()):
        if r["grid"] != g:
            raise SystemExit(f"{name}: grid {g} differs from the single-image run {r['grid']}")
    pad_count = sum(x == IMAGE_PAD for x in prompt_ids)
    if pad_count != sum(r["tokens"] for r in reports):
        raise SystemExit(f"{name}: {pad_count} image pads, expected {[r['tokens'] for r in reports]}")
    print(f"{name}: grids {grid.tolist()}, {len(prompt_ids)} prompt tokens", flush=True)

    im_end = tokenizer.convert_tokens_to_ids("<|im_end|>")
    eos = [im_end, tokenizer.convert_tokens_to_ids("<|endoftext|>")]
    generated = model.generate(**inputs, max_new_tokens=max_new, do_sample=False,
                               temperature=None, top_p=None, top_k=None,
                               eos_token_id=eos, pad_token_id=eos[1])
    continuation = generated[0, len(prompt_ids):].tolist()
    ids = prompt_ids + continuation
    P, L = len(prompt_ids), len(ids)
    text = tokenizer.decode(continuation)
    print(f"{name}: greedy {text!r}", flush=True)

    full = torch.tensor([ids])
    full_mm = (full == IMAGE_PAD).int()
    positions, delta = model.model.get_rope_index(
        full, mm_token_type_ids=full_mm, image_grid_thw=grid)
    positions = positions[:, 0, :]  # (3, L)
    rope_delta = int(delta.reshape(-1)[0])
    if int(positions.max()) + 1 - L != rope_delta:
        raise SystemExit(f"{name}: rope_delta mismatch")

    vision = model.model.get_image_features(inputs["pixel_values"], grid, return_dict=True)
    features = torch.cat(vision.pooler_output, dim=0).float()

    out_tf = model(input_ids=full, pixel_values=inputs["pixel_values"], image_grid_thw=grid,
                   mm_token_type_ids=full_mm, output_hidden_states=True, use_cache=False,
                   logits_to_keep=L - P + 1)
    logits = out_tf.logits[0].float().numpy()  # rows P-1 .. L-1
    hidden = torch.stack([h[0] for h in out_tf.hidden_states]).float().numpy()
    tf_argmax = logits[:-1].argmax(-1).tolist()
    match = sum(int(a == b) for a, b in zip(tf_argmax, continuation))

    # The same image resized by Pillow instead of torchvision: how far the
    # features move (what a C++ Pillow-bicubic reproduction would give).
    for i, (img, r) in enumerate(zip(images, reports)):
        if not r["resized"]:
            continue
        pil = Image.fromarray(pillow_resized_rgb(img, r["resized_height"],
                                                 r["resized_width"]).transpose(1, 2, 0))
        alt = processor.image_processor(images=[pil], size=SIZE, do_resize=False,
                                        return_tensors="pt")
        feats = model.model.get_image_features(alt["pixel_values"], alt["image_grid_thw"],
                                               return_dict=True).pooler_output[0].float()
        start = sum(x["tokens"] for x in reports[:i])
        ref = features[start:start + r["tokens"]]
        cos = F.cosine_similarity(ref, feats, dim=-1)
        rel = float((ref - feats).norm() / ref.norm())
        r["features_pillow_vs_hf"] = {"cosine_min": float(cos.min()),
                                      "cosine_mean": float(cos.mean()), "rel_l2": rel}

    files = {}
    (out / f"{name}.ids.json").write_text(json.dumps({"ids": ids, "prompt_len": P}))
    (out / f"{name}.grid.json").write_text(json.dumps(grid.tolist()))
    (out / f"{name}.greedy.json").write_text(json.dumps(continuation))
    files["positions.i32"] = write(out / f"{name}.positions.i32", positions.numpy(), np.int32)
    files["pixels.f32"] = write(out / f"{name}.pixels.f32", inputs["pixel_values"].numpy(), np.float32)
    files["features.f32"] = write(out / f"{name}.features.f32", features.numpy(), np.float32)
    files["logits.f32"] = write(out / f"{name}.logits.f32", logits, np.float32)
    files["hidden.f32"] = write(out / f"{name}.hidden.f32", hidden, np.float32)
    content_desc, k = [], 0
    for kind, value in content:
        if kind == "image":
            content_desc.append({"type": "image", "file": f"{name}.img{k}.png"})
            k += 1
        else:
            content_desc.append({"type": "text", "text": value})
    top = np.argsort(-logits[0])[:5].tolist()
    return {
        "content": content_desc,
        "prompt": prompt,
        "images": [dict(file=f"{name}.img{i}.png", **r) for i, r in enumerate(reports)],
        "image_tokens": pad_count,
        "prompt_len": P,
        "len": L,
        "continuation_len": L - P,
        "rope_delta": rope_delta,
        "logits_rows": [P - 1, L],
        "vocab": int(logits.shape[1]),
        "hidden_layers": int(hidden.shape[0]),
        "files": files,
        "greedy_text": text,
        "teacher_forced_argmax_match": [match, L - P],
        "first_top5": top,
        "first_top5_text": [tokenizer.decode([t]) for t in top],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--cases", nargs="*", help="run only these cases")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--resize-only", action="store_true",
                        help="only compare the processor's resize with Pillow's")
    args = parser.parse_args()
    torch.set_grad_enabled(False)
    torch.manual_seed(0)

    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(args.model)
    selected = [c for c in cases() if not args.cases or c[0] in args.cases]

    if args.resize_only:
        inputs = [(f"{n}.img{i}", v) for n, content in selected
                  for i, v in enumerate(x for k, x in content if k == "image")]
        inputs += [(n, page(w, h) if kind == "page" else gradient_card(w, h, 3))
                   for n, kind, w, h in SCAN_SIZES]
        report = {n: resize_report(processor, img) for n, img in inputs}
        print(json.dumps(report, indent=1))
        if args.out:
            args.out.write_text(json.dumps(report, indent=1) + "\n")
        return 0

    from transformers import Qwen3_5ForConditionalGeneration
    if args.out is None:
        parser.error("--out is required")
    args.out.mkdir(parents=True, exist_ok=True)
    tokenizer = processor.tokenizer
    model = Qwen3_5ForConditionalGeneration.from_pretrained(args.model, dtype=torch.float32)
    model.eval()
    manifest = {
        "model": args.model.name,
        "model_class": type(model).__name__,
        "transformers": transformers.__version__,
        "torch": torch.__version__,
        "pillow": PIL.__version__,
        "dtype": "float32",
        "image_processor": type(processor.image_processor).__name__,
        "size": SIZE,
        "max_new_tokens": args.max_new_tokens,
        "token_ids": {"vision_start": VISION_START, "vision_end": VISION_END,
                      "image_pad": IMAGE_PAD},
        "cases": {},
    }
    for name, content in selected:
        manifest["cases"][name] = run_case(name, content, model, processor, tokenizer,
                                           args.out, args.max_new_tokens)
        (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
