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
    "Qwen/Qwen2.5-VL-7B-Instruct",
    "Qwen/Qwen2.5-VL-3B-Instruct",
]

_INSTRUCTION = """You are an image analysis engine. Analyze the image and output ONE JSON object and nothing else, matching exactly this shape:
{
 "high_level_description": "<1-2 sentence summary of the whole image, naming the main subject and exactly what they are doing (their action/activity)>",
 "aesthetics": "<comma-separated visual mood/keywords>",
 "lighting": "<lighting description>",
 "photo": "<camera/lens e.g. '35mm, f/1.4'; empty string if not a photograph>",
 "art_style": "<art style if NOT a photograph; empty string otherwise>",
 "medium": "<one of: photograph, illustration, 3d_render, painting, digital_art>",
 "background": "<description of the environment/background>",
 "elements": [
   {"bbox": [x_min, y_min, x_max, y_max], "desc": "<detailed description>", "text": "<literal rendered text if this element is text, else empty string>"}
 ]
}
Rules:
- bbox values are integer PIXEL coordinates in the original image, which is {w} pixels wide and {h} pixels tall, top-left origin.
- List up to {n} of the most salient foreground elements, largest/most important first.
- Every 'desc' must be richly detailed and specific: materials, textures, colors, patterns, condition, lighting, and spatial relationships. Avoid vague one-word labels.{person}
- Output strictly valid JSON. No markdown fences, no commentary."""

# Used in hybrid (external SEGS) mode: global fields only, regions captioned separately.
_GLOBAL_INSTRUCTION = """You are an image analysis engine. Analyze the image and output ONE JSON object and nothing else, matching exactly this shape:
{
 "high_level_description": "<1-2 sentence summary naming the main subject and exactly what they are doing>",
 "aesthetics": "<comma-separated visual mood/keywords>",
 "lighting": "<lighting description>",
 "photo": "<camera/lens e.g. '35mm, f/1.4'; empty string if not a photograph>",
 "art_style": "<art style if NOT a photograph; empty string otherwise>",
 "medium": "<one of: photograph, illustration, 3d_render, painting, digital_art>",
 "background": "<description of the environment/background>"
}
Rules:
- Be specific and richly detailed.{person}
- Output strictly valid JSON. No markdown fences, no commentary."""

# Appended when anonymous=False: push fine-grained human detail.
_DETAIL_PERSON = (
    "\n- For any person, FIRST state what they are doing — the action/activity, any"
    " interaction with objects or other people, and whether the pose is static or"
    " in-motion. Then describe thoroughly: overall pose and body position; arms, hands"
    " and gesture; legs and feet/footwear; torso and build; face and facial expression;"
    " gaze/head direction; and hair (length, style, color). Note visible skin, tattoos,"
    " jewelry and other distinguishing features."
)

# Appended when detect_parts=True (non-anonymous): emit sub-part boxes.
_PARTS_FULL = (
    "\n- GRANULAR DETECTION: besides whole subjects, also output SEPARATE elements (each"
    " with its own bbox) for distinct visible sub-parts when clearly visible and not tiny:"
    " head/face, hair, each hand, each foot or shoe, and individual clothing items."
)

# Appended when detect_parts=True AND anonymous=True: clothing/accessory parts only.
_PARTS_ANON = (
    "\n- GRANULAR DETECTION: besides whole subjects, also output SEPARATE elements (each"
    " with its own bbox) for distinct clothing items, footwear, headwear, bags and"
    " accessories. Do NOT create elements for face, hair, skin or any body part."
)

# Appended when anonymous=True: hard ban on identifying/physical person detail.
_ANON_RULES = (
    "\n- PRIVACY MODE (STRICT): For any person you must NOT describe or even mention"
    " their face, eyes, gaze, hair, skin, complexion, body type, build, age, ethnicity,"
    " anatomy, tattoos, ink, body art, scars, birthmarks, piercings on skin, or ANY"
    " identifying or physical feature — even if clearly visible in the image. Refer to"
    " people only as 'a person', 'a woman', 'a man', 'she', 'he', or 'they'. Put all of"
    " your descriptive detail into their clothing, outfit, footwear, headwear, jewelry,"
    " accessories, held objects, and pose/action instead. This rule OVERRIDES every other"
    " detail instruction above. Apply it to high_level_description, background, and every"
    " element 'desc'."
)

# overlay colors (RGB) cycled per element
_OVERLAY_PALETTE = [
    (255, 59, 48), (52, 199, 89), (0, 122, 255), (255, 149, 0),
    (175, 82, 222), (255, 204, 0), (90, 200, 250), (255, 45, 85),
]


def _load_qwen(name, load_4bit=False):
    key = (name, load_4bit)
    if key in _MODELS:
        return _MODELS[key]
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    if load_4bit and device == "cuda":
        from transformers import BitsAndBytesConfig

        quant = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            name, dtype=dtype, quantization_config=quant, device_map="auto"
        ).eval()
    else:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            name, dtype=dtype
        ).to(device).eval()
    processor = AutoProcessor.from_pretrained(name)
    _MODELS[key] = (model, processor)
    return _MODELS[key]


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


def _region_prompt(label, anonymous):
    """Prompt to caption a single detected crop."""
    p = (
        "Describe what is shown in this cropped image in one vivid, detailed phrase"
        " (it was detected as '%s'). " % (label or "region")
    )
    if anonymous:
        p += (
            "Do NOT mention the face, eyes, hair, skin, body, tattoos or any identifying"
            " physical feature of a person; describe only clothing, accessories, footwear,"
            " held objects and pose/action. "
        )
    return p + "Reply with only the phrase — no quotes, no JSON, no label prefix."


def _caption_boxes(model, processor, pil_img, boxes, anonymous):
    """boxes: list of (xyxy_pixel, label). -> intermediate element dicts, each
    captioned by Qwen from its crop."""
    elements = []
    for xyxy, label in boxes:
        x1, y1, x2, y2 = (int(round(float(v))) for v in xyxy)
        x1, y1 = max(0, x1), max(0, y1)
        x2 = min(pil_img.width, x2)
        y2 = min(pil_img.height, y2)
        if x2 <= x1 or y2 <= y1:
            continue
        crop = pil_img.crop((x1, y1, x2, y2))
        desc = _run_qwen(
            model, processor, crop, _region_prompt(label, anonymous), max_new_tokens=160
        ).strip()
        elements.append({"bbox": [x1, y1, x2, y2], "desc": desc, "text": ""})
    return elements


# ---- ultralytics YOLO detectors -------------------------------------------
_YOLO_FOLDERS = ("ultralytics_bbox", "ultralytics_segm")
_YOLO_MODELS = {}  # path -> YOLO


def _yolo_paths():
    """Map of available detector filename -> full path, via ComfyUI folder_paths."""
    try:
        import folder_paths
    except Exception:
        return {}
    out = {}
    for kind in _YOLO_FOLDERS:
        try:
            for name in folder_paths.get_filename_list(kind):
                out.setdefault(name, folder_paths.get_full_path(kind, name))
        except Exception:
            pass
    return out


def _yolo_choices():
    return ["(none)"] + sorted(_yolo_paths().keys())


def _load_yolo(path):
    if path not in _YOLO_MODELS:
        from ultralytics import YOLO

        _YOLO_MODELS[path] = YOLO(path)
    return _YOLO_MODELS[path]


def _detect_boxes(pil_img, detector_names, conf):
    """Run each selected detector; return [(xyxy_pixel, label), ...] sorted by confidence."""
    paths = _yolo_paths()
    found = []
    for name in detector_names:
        path = paths.get(name)
        if not path:
            continue
        res = _load_yolo(path).predict(pil_img, conf=conf, verbose=False)[0]
        names = getattr(res, "names", {}) or {}
        stem = name.rsplit("/", 1)[-1].rsplit("\\", 1)[-1].rsplit(".", 1)[0]
        for b in res.boxes:
            xyxy = [float(v) for v in b.xyxy[0].tolist()]
            ci = int(b.cls[0]) if b.cls is not None and len(b.cls) else -1
            label = names.get(ci) if isinstance(names, dict) else None
            conf_v = float(b.conf[0]) if b.conf is not None and len(b.conf) else 0.0
            found.append((xyxy, label or stem, conf_v))
    found.sort(key=lambda t: t[2], reverse=True)
    return [(xyxy, label) for xyxy, label, _ in found]


# ---- helpers ---------------------------------------------------------------
def _tensor_to_pil(image):
    """ComfyUI IMAGE [B,H,W,C] float 0-1 -> first-frame PIL RGB."""
    arr = (image[0].cpu().numpy() * 255.0).round().clip(0, 255).astype(np.uint8)
    return Image.fromarray(arr, "RGB")


def _balance_json(frag):
    """Append the closing brackets needed to balance `frag` (ignoring strings).
    Returns None if the fragment ends inside an open string (unsafe cut)."""
    stack, in_str, esc = [], False, False
    for ch in frag:
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch in "{[":
            stack.append(ch)
        elif ch in "}]" and stack:
            stack.pop()
    if in_str:
        return None
    closers = "".join("}" if c == "{" else "]" for c in reversed(stack))
    return frag + closers


def _parse_json(raw):
    """Parse the model's JSON, tolerating fences, trailing prose, trailing commas,
    and truncated/corrupted tails (salvages by trimming to the last valid element)."""
    s = raw.strip()
    s = re.sub(r"^```(?:json)?\s*|\s*```$", "", s, flags=re.MULTILINE).strip()
    start = s.find("{")
    if start == -1:
        raise ValueError("no JSON object in model response:\n" + raw[:500])
    s = s[start:]
    dec = json.JSONDecoder()

    # 1) direct (raw_decode ignores any trailing junk after the object)
    for cand in (s, re.sub(r",(\s*[}\]])", r"\1", s)):  # also try w/o trailing commas
        try:
            return dec.raw_decode(cand)[0]
        except json.JSONDecodeError:
            pass

    # 2) salvage: from the end, cut at each element/array boundary, balance the
    #    open brackets, and try to parse. Drops a truncated or quote-corrupted tail
    #    while keeping the earlier complete elements. A ',' cut drops the whole
    #    broken element; a '}'/']' cut keeps it.
    for cut in range(len(s) - 1, -1, -1):
        ch = s[cut]
        if ch in "}]":
            frag = s[: cut + 1]
        elif ch == ",":
            frag = s[:cut]
        else:
            continue
        balanced = _balance_json(frag)
        if balanced is None:
            continue
        try:
            return dec.raw_decode(balanced)[0]
        except json.JSONDecodeError:
            continue
    raise ValueError("could not parse model JSON:\n" + raw[:1000])


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


def _gen_size(w, h, base):
    """Generation size preserving source aspect ratio: longer side = base,
    both rounded to a multiple of 64, min 512 (diffusion-friendly)."""
    base = max(512, (int(base) // 64) * 64)
    if w >= h:
        gw, gh = base, int(round(base * h / w / 64)) * 64
    else:
        gw, gh = int(round(base * w / h / 64)) * 64, base
    return max(512, gw), max(512, gh)


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
                "detect_parts": ("BOOLEAN", {"default": False}),
                "gen_base_size": ("INT", {"default": 1280, "min": 512, "max": 4096, "step": 64}),
                "load_4bit": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "detector_1": (_yolo_choices(),),
                "detector_2": (_yolo_choices(),),
                "detector_3": (_yolo_choices(),),
                "detector_4": (_yolo_choices(),),
                "yolo_confidence": ("FLOAT", {"default": 0.35, "min": 0.05, "max": 1.0, "step": 0.05}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "IMAGE", "INT", "INT")
    RETURN_NAMES = ("json_prompt", "preview", "bbox_overlay", "gen_width", "gen_height")
    FUNCTION = "generate"
    CATEGORY = "Ideogram4"
    OUTPUT_NODE = True

    def generate(self, image, vlm_model, max_elements, include_colors, anonymous,
                 detect_parts, gen_base_size, load_4bit,
                 detector_1="(none)", detector_2="(none)", detector_3="(none)",
                 detector_4="(none)", yolo_confidence=0.35):
        pil = _tensor_to_pil(image)
        gen_w, gen_h = _gen_size(pil.width, pil.height, gen_base_size)
        model, processor = _load_qwen(vlm_model, load_4bit)

        person = _ANON_RULES if anonymous else _DETAIL_PERSON
        if detect_parts:
            person += _PARTS_ANON if anonymous else _PARTS_FULL

        detectors = [d for d in (detector_1, detector_2, detector_3, detector_4)
                     if d and d != "(none)"]
        if detectors:
            # Hybrid: YOLO finds the boxes, Qwen captions each + the global fields.
            g_instr = _GLOBAL_INSTRUCTION.replace("{person}", person)
            data = _parse_json(_run_qwen(model, processor, pil, g_instr, max_new_tokens=1024))
            boxes = _detect_boxes(pil, detectors, yolo_confidence)[:max_elements]
            data["elements"] = _caption_boxes(model, processor, pil, boxes, anonymous)
        else:
            instruction = (
                _INSTRUCTION.replace("{n}", str(max(max_elements, 1)))
                .replace("{w}", str(pil.width))
                .replace("{h}", str(pil.height))
                .replace("{person}", person)
            )
            raw = _run_qwen(model, processor, pil, instruction, max_new_tokens=2560)
            data = _parse_json(raw)

        result = _build_id4(data, pil, max_elements, include_colors)
        overlay = _pil_to_tensor(
            _draw_overlay(pil, result["compositional_deconstruction"]["elements"])
        )
        compact = json.dumps(result, separators=(",", ":"), ensure_ascii=False)
        pretty = json.dumps(result, indent=2, ensure_ascii=False)
        return {
            "ui": {"text": [pretty]},
            "result": (compact, pretty, overlay, gen_w, gen_h),
        }


NODE_CLASS_MAPPINGS = {"Id4JsonPromptFromImage": Id4JsonPromptFromImage}
NODE_DISPLAY_NAME_MAPPINGS = {
    "Id4JsonPromptFromImage": "Ideogram-4 JSON Prompt from Image"
}
