"""ComfyUI node: image -> Qwen2.5-VL -> Ideogram-4 JSON prompt.

One VLM pass returns an intermediate JSON (global desc, style fields, background,
and a list of elements with normalized 0-1000 bboxes). Python then maps that to the
strict Ideogram-4 caption schema (key ordering, photo/art_style branch, bbox reorder
to [y_min,x_min,y_max,x_max], per-element color palettes, compact serialization).

Qwen2.5-VL lives in transformers core (no remote-code / timm / wandb baggage).
"""

import json
import re

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

# ---- model cache -----------------------------------------------------------
_MODELS = {}  # name -> (model, processor)

QWEN_CHOICES = [
    "Qwen/Qwen2.5-VL-3B-Instruct",
    "Qwen/Qwen2.5-VL-7B-Instruct",
]

_INSTRUCTION = """You are an image analysis engine. Analyze the image and output ONE JSON object and nothing else, matching exactly this shape:
{
 "high_level_description": "<1-2 sentence summary of the whole image>",
 "aesthetics": "<comma-separated visual mood/keywords>",
 "lighting": "<lighting description>",
 "photo": "<camera/lens e.g. '35mm, f/1.4'; empty string if not a photograph>",
 "art_style": "<art style if NOT a photograph; empty string otherwise>",
 "medium": "<one of: photograph, illustration, 3d_render, painting, digital_art>",
 "background": "<description of the environment/background>",
 "elements": [
   {"bbox": [x_min, y_min, x_max, y_max], "desc": "<short description>", "text": "<literal rendered text if this element is text, else empty string>"}
 ]
}
Rules:
- bbox values are integer PIXEL coordinates in the original image, which is {w} pixels wide and {h} pixels tall, top-left origin.
- List up to {n} of the most salient foreground elements, largest/most important first.{anon}
- Output strictly valid JSON. No markdown fences, no commentary."""

_ANON_RULES = (
    "\n- PRIVACY MODE: Do NOT describe identifying or physical attributes of any person"
    " — no face, eyes, hair (style/length/color), skin, body type, age, ethnicity,"
    " tattoos, scars or distinguishing marks. Refer to people only generically (a woman,"
    " a man, a person, she, he, they). DO still describe clothing, outfit, footwear,"
    " jewelry, accessories and pose/action. Apply this to high_level_description,"
    " background, and every element 'desc'."
)

# overlay colors (RGB) cycled per element
_OVERLAY_PALETTE = [
    (255, 59, 48), (52, 199, 89), (0, 122, 255), (255, 149, 0),
    (175, 82, 222), (255, 204, 0), (90, 200, 250), (255, 45, 85),
]


def _load_qwen(name):
    if name in _MODELS:
        return _MODELS[name]
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        name, dtype=dtype
    ).to(device).eval()
    processor = AutoProcessor.from_pretrained(name)
    _MODELS[name] = (model, processor)
    return _MODELS[name]


def _run_qwen(model, processor, pil_img, instruction, max_new_tokens=1024):
    device = model.device
    messages = [
        {
            "role": "user",
            "content": [{"type": "image"}, {"type": "text", "text": instruction}],
        }
    ]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(text=[text], images=[pil_img], return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    trimmed = out[:, inputs.input_ids.shape[1] :]
    return processor.batch_decode(
        trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]


# ---- helpers ---------------------------------------------------------------
def _tensor_to_pil(image):
    """ComfyUI IMAGE [B,H,W,C] float 0-1 -> first-frame PIL RGB."""
    arr = (image[0].cpu().numpy() * 255.0).round().clip(0, 255).astype(np.uint8)
    return Image.fromarray(arr, "RGB")


def _parse_json(raw):
    """Pull the first JSON object out of a model response (tolerates fences/prose)."""
    s = raw.strip()
    s = re.sub(r"^```(?:json)?|```$", "", s, flags=re.MULTILINE).strip()
    start, end = s.find("{"), s.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("no JSON object in model response:\n" + raw[:500])
    return json.loads(s[start : end + 1])


def _pixel_xyxy_to_id4(bbox, w, h):
    """pixel [x_min,y_min,x_max,y_max] -> normalized 0-1000 [y_min,x_min,y_max,x_max]."""
    x0, y0, x1, y1 = (float(v) for v in bbox)
    nx0, nx1 = sorted((x0 / w * 1000, x1 / w * 1000))
    ny0, ny1 = sorted((y0 / h * 1000, y1 / h * 1000))
    clamp = lambda v: max(0, min(1000, int(round(v))))
    return [clamp(ny0), clamp(nx0), clamp(ny1), clamp(nx1)]


def _dominant_colors(pil_img, n=5):
    """Up to n dominant colors as uppercase #RRGGBB, most-frequent first."""
    if pil_img.width < 1 or pil_img.height < 1:
        return []
    q = pil_img.convert("RGB").quantize(colors=n, method=Image.FASTOCTREE)
    pal = q.getpalette()
    out = []
    for _, idx in sorted(q.getcolors(), reverse=True):
        r, g, b = pal[idx * 3 : idx * 3 + 3]
        out.append("#{:02X}{:02X}{:02X}".format(r, g, b))
    return out


def _pil_to_tensor(pil_img):
    """PIL RGB -> ComfyUI IMAGE [1,H,W,3] float 0-1."""
    arr = np.asarray(pil_img.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(arr)[None, ...]


def _draw_overlay(pil_img, elements):
    """Draw each element's id4 bbox + a numbered label onto a copy of the image."""
    img = pil_img.convert("RGB").copy()
    d = ImageDraw.Draw(img)
    w, h = img.size
    try:
        font = ImageFont.truetype("arial.ttf", max(12, w // 60))
    except Exception:
        font = ImageFont.load_default()
    line = max(2, w // 320)
    for i, el in enumerate(elements):
        y0, x0, y1, x1 = el["bbox"]
        px0, py0 = x0 / 1000 * w, y0 / 1000 * h
        px1, py1 = x1 / 1000 * w, y1 / 1000 * h
        color = _OVERLAY_PALETTE[i % len(_OVERLAY_PALETTE)]
        d.rectangle([px0, py0, px1, py1], outline=color, width=line)
        label = "{}: {}".format(i, (el.get("text") or el.get("desc") or "")[:28])
        tb = d.textbbox((px0, py0), label, font=font)
        d.rectangle([tb[0] - 1, tb[1] - 1, tb[2] + 2, tb[3] + 2], fill=color)
        d.text((px0, py0), label, fill=(255, 255, 255), font=font)
    return img


def _crop_from_id4_bbox(pil_img, id4_bbox):
    """[y_min,x_min,y_max,x_max] 0-1000 -> pixel crop of pil_img."""
    y0, x0, y1, x1 = id4_bbox
    w, h = pil_img.width, pil_img.height
    return pil_img.crop(
        (round(x0 / 1000 * w), round(y0 / 1000 * h),
         round(x1 / 1000 * w), round(y1 / 1000 * h))
    )


def _build_id4(data, pil_img, max_elements, include_colors):
    """Map intermediate model JSON -> strict Ideogram-4 schema dict."""
    w, h = pil_img.width, pil_img.height
    medium = (data.get("medium") or "photograph").strip()
    is_photo = medium == "photograph" or bool((data.get("photo") or "").strip())

    style = {"aesthetics": (data.get("aesthetics") or "").strip(),
             "lighting": (data.get("lighting") or "").strip()}
    if is_photo:
        style["photo"] = (data.get("photo") or "").strip()
        style["medium"] = medium
    else:
        style["medium"] = medium
        style["art_style"] = (data.get("art_style") or "").strip()
    if include_colors:
        style["color_palette"] = _dominant_colors(pil_img, 16)

    elements = []
    for el in (data.get("elements") or [])[:max_elements]:
        bbox = el.get("bbox")
        if not (isinstance(bbox, (list, tuple)) and len(bbox) == 4):
            continue
        id4_bbox = _pixel_xyxy_to_id4(bbox, w, h)
        text = (el.get("text") or "").strip()
        desc = (el.get("desc") or "").strip()
        out = {"type": "text" if text else "obj", "bbox": id4_bbox}
        if text:
            out["text"] = text
        out["desc"] = desc
        if include_colors:
            pal = _dominant_colors(_crop_from_id4_bbox(pil_img, id4_bbox), 5)
            if pal:
                out["color_palette"] = pal
        elements.append(out)

    return {
        "high_level_description": (data.get("high_level_description") or "").strip(),
        "style_description": style,
        "compositional_deconstruction": {
            "background": (data.get("background") or "").strip(),
            "elements": elements,
        },
    }


# ---- node ------------------------------------------------------------------
class Id4JsonPromptFromImage:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "vlm_model": (QWEN_CHOICES,),
                "max_elements": ("INT", {"default": 8, "min": 0, "max": 64}),
                "include_colors": ("BOOLEAN", {"default": True}),
                "anonymous": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "IMAGE")
    RETURN_NAMES = ("json_prompt", "preview", "bbox_overlay")
    FUNCTION = "generate"
    CATEGORY = "Ideogram4"
    OUTPUT_NODE = True

    def generate(self, image, vlm_model, max_elements, include_colors, anonymous):
        pil = _tensor_to_pil(image)
        model, processor = _load_qwen(vlm_model)

        instruction = (
            _INSTRUCTION.replace("{n}", str(max(max_elements, 1)))
            .replace("{w}", str(pil.width))
            .replace("{h}", str(pil.height))
            .replace("{anon}", _ANON_RULES if anonymous else "")
        )
        raw = _run_qwen(model, processor, pil, instruction)
        data = _parse_json(raw)

        result = _build_id4(data, pil, max_elements, include_colors)
        overlay = _pil_to_tensor(
            _draw_overlay(pil, result["compositional_deconstruction"]["elements"])
        )
        compact = json.dumps(result, separators=(",", ":"), ensure_ascii=False)
        pretty = json.dumps(result, indent=2, ensure_ascii=False)
        return {"ui": {"text": [pretty]}, "result": (compact, pretty, overlay)}


NODE_CLASS_MAPPINGS = {"Id4JsonPromptFromImage": Id4JsonPromptFromImage}
NODE_DISPLAY_NAME_MAPPINGS = {
    "Id4JsonPromptFromImage": "Ideogram-4 JSON Prompt from Image"
}
