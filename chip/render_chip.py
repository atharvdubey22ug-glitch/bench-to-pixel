#!/usr/bin/env python3
"""Render full_adder_chip.gds to PNG (KLayout, headless) with labels."""
import os
import sys

import klayout.db as db
import klayout.lay as lay
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
COL = {(1, 0): 0xff5a47, (11, 0): 0xeba340, (12, 0): 0x8a6f3a, (13, 0): 0x8fe3ff, (99, 0): 0x33464f}


def render(gds, png, w=2400, h=1680, labels=True):
    v = lay.LayoutView()
    v.load_layout(gds, True)
    v.max_hier()
    v.set_config("background-color", "#0a1013")
    v.set_config("grid-visible", "false")
    v.set_config("text-visible", "false")
    it = v.begin_layers()
    while not it.at_end():
        lp = it.current()
        key = (lp.source_layer, lp.source_datatype)
        lp.visible = key in COL
        if key in COL:
            lp.fill_color = lp.frame_color = COL[key]
            lp.dither_pattern = {(1, 0): 0, (11, 0): 0, (12, 0): 0, (13, 0): 1, (99, 0): 1}[key]
            lp.width = 1
        v.set_layer_properties(it, lp)
        it.next()
    ly = db.Layout()
    ly.read(gds)
    bb0 = ly.top_cell().dbbox()
    bb = db.DBox(bb0.left - 330, bb0.bottom - 40, bb0.right + 40, bb0.top + 40)
    v.zoom_box(bb)
    v.save_image(png, w, h)
    if not labels:
        return
    im = Image.open(png).convert("RGB")
    d = ImageDraw.Draw(im)
    try:
        f = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 22)
    except OSError:
        f = ImageFont.load_default()
    sx = w / bb.width()
    sy = h / bb.height()
    s = min(sx, sy)
    ox = (w - bb.width() * s) / 2
    oy = (h - bb.height() * s) / 2

    def px(x, y):
        return ox + (x - bb.left) * s, oy + (bb.top - y) * s

    for y, t in ((635, "loopback"), (508, "loopback"), (381, "XOR1 -> comparator -> X"), (254, "carry b (B)"), (127, "carry a (A)"),
                 (0, "LASER IN"), (-127, "carry c (Cin)"), (-254, "carry d (X)"), (-381, "SUM")):
        X, Y = px(-45, y)
        d.text((X - 10, Y - 12), t, fill=(227, 235, 237), font=f, anchor="rm")
    for (x, y), t in (((1450, 70), "Michelson XOR 1 (A, B)  ^"), ((1450, -70), "Michelson XOR 2 (Cin, X)  v"),
                      ((560, 200), "AND gate 1 switches"), ((560, -200), "AND gate 2 switches"),
                      ((450, 680), "heater pads, block 1"), ((450, -680), "heater pads, block 2")):
        X, Y = px(x, y)
        d.text((X, Y), t, fill=(235, 163, 64), font=f, anchor="mm")
    im.save(png)


if __name__ == "__main__":
    gds = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "full_adder_chip.gds")
    out = sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, "figures", "chip_layout.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    render(gds, out)
    print(out)
