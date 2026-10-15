#!/usr/bin/env python3
"""Render the KLayout layout (klayout/EBeam_heaters_atharvdubey_FullAdder.gds) to PNG with labels."""
import os
import sys

import klayout.db as db
import klayout.lay as lay
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
COL = {(998, 0): 0xff5a47, (1, 0): 0xff5a47, (11, 0): 0xeba340, (12, 0): 0x8a6f3a, (13, 0): 0x8fe3ff, (99, 0): 0x33464f}


def render(gds, png, w=2400, h=1840, labels=True):
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
            lp.dither_pattern = {(998, 0): 0, (1, 0): 0, (11, 0): 0, (12, 0): 0, (13, 0): 1, (99, 0): 1}[key]
            lp.width = 1
        v.set_layer_properties(it, lp)
        it.next()
    ly = db.Layout()
    ly.read(gds)
    bb0 = ly.top_cell().dbbox()
    bb = db.DBox(bb0.left - 300, bb0.bottom - 20, bb0.right + 20, bb0.top + 20)
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

    S = (120.0, 860.0)          # design -> GDS shift used by klayout/full_adder_siepic.py
    for y, t in ((635, "loopback"), (508, "loopback"), (381, "XOR 1 -> comparator -> X"), (254, "LASER 1 in"),
                 (127, "carry a (A)"), (0, "carry b (B)"), (-381, "SUM"), (-508, "LASER 2 in"),
                 (-635, "carry c (Cin)"), (-762, "carry d (X)")):
        X, Y = px(S[0] - 45, S[1] + y)
        d.text((X - 10, Y - 12), t, fill=(227, 235, 237), font=f, anchor="rm")
    for (x, y), t in (((1450, 160), "half adder 1: Michelson XOR (A, B)"), ((1450, -600), "half adder 2: Michelson XOR (Cin, X)"),
                      ((560, 75), "AND gate 1 switches"), ((560, -690), "AND gate 2 switches"),
                      ((450, 680), "heater pads, block 1"), ((450, -80), "heater pads, block 2")):
        X, Y = px(S[0] + x, S[1] + y)
        d.text((X, Y), t, fill=(235, 163, 64), font=f, anchor="mm")
    im.save(png)


if __name__ == "__main__":
    gds = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "klayout", "EBeam_heaters_atharvdubey_FullAdder.gds")
    out = sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, "figures", "chip_layout.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    render(gds, out)
    print(out)
