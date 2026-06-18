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
    # tolerates trailing prose + trailing commas
    assert nodes._parse_json('{"a":1,}\nhere you go!')["a"] == 1
    # salvages a TRUNCATED tail, keeping earlier complete elements
    truncated = '{"elements":[{"desc":"a"},{"desc":"b"},{"desc":"c'
    got = nodes._parse_json(truncated)
    assert [e["desc"] for e in got["elements"]] == ["a", "b"]
    # salvages a tail corrupted by an unescaped quote
    bad = '{"high_level_description":"ok","elements":[{"desc":"sign reads "OPEN" loud"}]}'
    assert nodes._parse_json(bad)["high_level_description"] == "ok"

    # person block swaps between detail and strict-privacy
    assert "{person}" in nodes._INSTRUCTION
    assert "PRIVACY MODE" in nodes._ANON_RULES and "tattoos" in nodes._ANON_RULES
    assert "hands" in nodes._DETAIL_PERSON and "hair" in nodes._DETAIL_PERSON
    # parts blocks: full mentions body parts, anon variant forbids them
    assert "foot or shoe" in nodes._PARTS_FULL
    assert "Do NOT create elements for face" in nodes._PARTS_ANON

    # overlay draws without error and returns a same-size IMAGE tensor
    ov = nodes._draw_overlay(
        Image.new("RGB", (120, 90)),
        [{"bbox": [100, 200, 900, 800], "desc": "x"}],
    )
    assert nodes._pil_to_tensor(ov).shape[1:3] == (90, 120)

    # gen size preserves AR, multiples of 64, longer side = base
    assert nodes._gen_size(1000, 1000, 1280) == (1280, 1280)
    gw, gh = nodes._gen_size(1600, 900, 1280)  # 16:9 landscape
    assert gw == 1280 and gh % 64 == 0 and abs(gw / gh - 1600 / 900) < 0.05
    gw, gh = nodes._gen_size(900, 1600, 1280)  # 9:16 portrait -> longer = height
    assert gh == 1280 and gw % 64 == 0
    print("helpers OK")


def test_generate_mocked(monkeypatch):
    monkeypatch.setattr(nodes, "_load_qwen", lambda *a, **k: (object(), object()))

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
    compact, _, overlay, gw, gh = node.generate(
        img, "Qwen/Qwen2.5-VL-7B-Instruct", 8, True, False, False, 1280, False
    )["result"]
    obj = json.loads(compact)
    # overlay is a ComfyUI IMAGE tensor [1,H,W,3] matching the input size
    assert tuple(overlay.shape) == (1, 100, 100, 3)
    # square input -> square gen size
    assert (gw, gh) == (1280, 1280)

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


def test_hybrid_mocked(monkeypatch):
    # global call returns style/background JSON; region calls return a phrase
    def fake_run(model, processor, img, instruction, max_new_tokens=1024):
        if "ONE JSON object" in instruction:
            return json.dumps({
                "high_level_description": "two people standing",
                "aesthetics": "", "lighting": "", "photo": "",
                "art_style": "", "medium": "photograph", "background": "a street",
            })
        return "a detailed region phrase"

    monkeypatch.setattr(nodes, "_load_qwen", lambda *a, **k: (object(), object()))
    monkeypatch.setattr(nodes, "_run_qwen", fake_run)
    # YOLO returns two boxes; skip real ultralytics
    monkeypatch.setattr(
        nodes, "_detect_boxes",
        lambda pil, dets, conf: [([10, 10, 40, 40], "face"), ([50, 60, 80, 90], "hand")],
    )

    node = nodes.Id4JsonPromptFromImage()
    img = torch.zeros(1, 100, 100, 3)
    compact, _, _, _, _ = node.generate(
        img, "Qwen/Qwen2.5-VL-7B-Instruct", 8, True, False, False, 1280, False,
        "face_yolov8m.pt", "hand_yolov8s.pt", "(none)", "(none)", 0.35,
    )["result"]
    obj = json.loads(compact)
    assert obj["high_level_description"] == "two people standing"
    els = obj["compositional_deconstruction"]["elements"]
    assert len(els) == 2 and els[0]["bbox"] == [100, 100, 400, 400]
    assert els[0]["desc"] == "a detailed region phrase"
    print("hybrid (mocked) OK")


if __name__ == "__main__":
    test_helpers()

    class _MP:
        def setattr(self, obj, name, val):
            setattr(obj, name, val)

    test_generate_mocked(_MP())
    test_hybrid_mocked(_MP())
    print("ALL PASS")
