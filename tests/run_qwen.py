"""Tier-2: real Qwen2.5-VL end-to-end. Downloads model on first run (~7GB for 3B)."""
import json
import os
import sys

import numpy as np
import torch
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import nodes


def make_image():
    """Simple synthetic scene: sky, sun, a house + tree."""
    img = Image.new("RGB", (640, 480), (135, 206, 235))
    d = ImageDraw.Draw(img)
    d.rectangle([0, 360, 640, 480], fill=(34, 139, 34))      # grass
    d.ellipse([520, 40, 600, 120], fill=(255, 215, 0))        # sun
    d.rectangle([180, 240, 360, 380], fill=(178, 34, 34))     # house
    d.polygon([(180, 240), (360, 240), (270, 160)], fill=(110, 38, 14))  # roof
    d.rectangle([430, 280, 460, 380], fill=(101, 67, 33))     # trunk
    d.ellipse([390, 200, 500, 300], fill=(0, 100, 0))         # foliage
    return img


def main():
    pil = make_image()
    arr = np.asarray(pil, dtype=np.float32) / 255.0
    image = torch.from_numpy(arr)[None, ...]  # [1,H,W,3]

    node = nodes.Id4JsonPromptFromImage()
    model_name = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen2.5-VL-7B-Instruct"
    load_4bit = len(sys.argv) > 2 and sys.argv[2] == "4bit"
    compact, pretty, overlay, gw, gh = node.generate(
        image, model_name, 8, True, False, False, 1280, load_4bit
    )["result"]
    print(pretty)
    print("gen size:", gw, "x", gh)

    obj = json.loads(compact)
    assert "compositional_deconstruction" in obj
    for el in obj["compositional_deconstruction"]["elements"]:
        y0, x0, y1, x1 = el["bbox"]
        assert 0 <= y0 <= y1 <= 1000 and 0 <= x0 <= x1 <= 1000, el["bbox"]

    # save the bbox overlay for eyeballing
    ov = (overlay[0].numpy() * 255).round().astype(np.uint8)
    out = os.path.join(os.path.dirname(__file__), "overlay_out.png")
    Image.fromarray(ov, "RGB").save(out)
    print("\nElements:", len(obj["compositional_deconstruction"]["elements"]))
    print("overlay saved:", out)
    print("TIER-2 OK")


if __name__ == "__main__":
    main()
