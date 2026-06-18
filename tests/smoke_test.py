"""Tier-1 smoke test: helpers + JSON mapping, Qwen2.5-VL mocked (no download)."""
import json
import os
import sys

import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import nodes


def test_helpers():
    # bbox: pixel [x_min,y_min,x_max,y_max] -> normalized 0-1000 [y_min,x_min,y_max,x_max]
    assert nodes._pixel_xyxy_to_id4([300, 100, 900, 700], 1000, 1000) == [100, 300, 700, 900]
    assert nodes._pixel_xyxy_to_id4([50, 100, 100, 200], 100, 200) == [500, 500, 1000, 1000]
    # clamps + orders reversed coords
    assert nodes._pixel_xyxy_to_id4([1200, -5, 50, 80], 1000, 100) == [0, 50, 800, 1000]

    # tensor -> PIL
    img = torch.zeros(1, 100, 100, 3)
    img[0, :, :, 0] = 1.0  # full red
    pil = nodes._tensor_to_pil(img)
    assert pil.size == (100, 100) and pil.getpixel((0, 0)) == (255, 0, 0)

    # colors -> uppercase #RRGGBB
    assert nodes._dominant_colors(pil, 5)[0] == "#FF0000"

    # JSON parse tolerates fences
    assert nodes._parse_json('```json\n{"a": 1}\n```')["a"] == 1

    # anonymity rules injected only when requested
    assert "{anon}" not in nodes._INSTRUCTION.replace("{anon}", nodes._ANON_RULES)
    assert "PRIVACY MODE" in nodes._ANON_RULES

    # overlay draws without error and returns a same-size IMAGE tensor
    ov = nodes._draw_overlay(
        Image.new("RGB", (120, 90)),
        [{"bbox": [100, 200, 900, 800], "desc": "x"}],
    )
    assert nodes._pil_to_tensor(ov).shape[1:3] == (90, 120)
    print("helpers OK")


def test_generate_mocked(monkeypatch):
    monkeypatch.setattr(nodes, "_load_qwen", lambda name: (object(), object()))

    fake = json.dumps({
        "high_level_description": "A red square test image.",
        "aesthetics": "minimal, bold",
        "lighting": "flat studio light",
        "photo": "",
        "art_style": "flat vector",
        "medium": "illustration",
        "background": "a plain neutral backdrop",
        "elements": [
            {"bbox": [20, 20, 60, 60], "desc": "a red region", "text": ""},
            {"bbox": [0, 0, 100, 10], "desc": "title", "text": "HELLO"},
        ],
    })
    monkeypatch.setattr(nodes, "_run_qwen", lambda *a, **k: fake)

    node = nodes.Id4JsonPromptFromImage()
    img = torch.zeros(1, 100, 100, 3)  # 100x100 -> pixel coords map 1:10 to 0-1000
    img[0, :, :, 0] = 1.0
    compact, _, overlay = node.generate(
        img, "Qwen/Qwen2.5-VL-3B-Instruct", 8, True, False
    )["result"]
    obj = json.loads(compact)
    # overlay is a ComfyUI IMAGE tensor [1,H,W,3] matching the input size
    assert tuple(overlay.shape) == (1, 100, 100, 3)

    assert obj["high_level_description"] == "A red square test image."
    # non-photo -> art_style branch, no 'photo' key
    sd = obj["style_description"]
    assert "art_style" in sd and "photo" not in sd
    assert sd["medium"] == "illustration"

    els = obj["compositional_deconstruction"]["elements"]
    assert els[0]["type"] == "obj" and els[0]["bbox"] == [200, 200, 600, 600]
    assert els[1]["type"] == "text" and els[1]["text"] == "HELLO"
    assert els[1]["bbox"] == [0, 0, 100, 1000]
    # strict key order for text element: type, bbox, text, desc, (color_palette)
    keys = list(els[1].keys())
    assert keys[:4] == ["type", "bbox", "text", "desc"], keys
    # compact serialization: no spaces after STRUCTURAL separators
    assert '"style_description":{' in compact
    assert '"compositional_deconstruction":{"background":' in compact
    print("generate (mocked) OK")


if __name__ == "__main__":
    test_helpers()

    class _MP:
        def setattr(self, obj, name, val):
            setattr(obj, name, val)

    test_generate_mocked(_MP())
    print("ALL PASS")
