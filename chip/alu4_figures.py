#!/usr/bin/env python3
"""Figures for the 4-bit ALU: layout render and the decision levels of every bit (stressed die)."""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
INK, MUT, FG, ONE, ZERO = "#0a1013", "#8b9ea5", "#e3ebed", "#2c9fd6", "#e5503f"
plt.rcParams.update({"figure.facecolor": INK, "axes.facecolor": INK, "axes.edgecolor": "#33464f", "axes.labelcolor": MUT,
                     "xtick.color": MUT, "ytick.color": MUT, "text.color": FG, "font.size": 10,
                     "axes.grid": True, "grid.color": "#1c2a31"})


def levels(die="stressed"):
    eyes = json.load(open(os.path.join(HERE, "alu4_eyes.json")))[die]
    res = json.load(open(os.path.join(HERE, "alu4_results.json")))[die]
    fig, axs = plt.subplots(1, 3, figsize=(11.5, 3.8))
    rng = np.random.default_rng(0)
    for ax, (k, name) in zip(axs, (("xor1", "XOR 1"), ("sum", "SUM"), ("carry", "CARRY"))):
        for i in range(4):
            e = eyes[f"bit{i}_{k}"]
            for vals, col, fill, dx in ((e["ones"], ONE, True, -0.14), (e["zeros"], ZERO, False, 0.14)):
                v = np.unique(np.round(vals, 2))
                x = i + dx + rng.uniform(-0.06, 0.06, v.size)
                ax.scatter(x, v, s=18, facecolors=col if fill else "none", edgecolors=col, linewidths=1.2, zorder=3)
            ax.hlines(e["threshold"], i - 0.32, i + 0.32, color=FG, lw=1.5, zorder=4)
        q = min(res["decisions"][f"bit{i}_{k}"]["margin_pct"] for i in range(4))
        ax.set_ylim(bottom=0)
        ax.set_xticks(range(4), [f"bit {i}" for i in range(4)])
        ax.set_title(f"{name}: worst margin {q:.0f}%", color=FG, fontsize=10)
        ax.grid(axis="x", visible=False)
    axs[0].set_ylabel("photocurrent (uA), 1 mW per bit")
    fig.text(0.5, 0.045, "filled blue: decisions that should read 1      hollow red: should read 0      white bar: that bit's comparator threshold",
             ha="center", color=MUT, fontsize=9)
    fig.text(0.5, 0.01, "every operand pair of ADD, SUB, AND, OR and XOR, with each bit at a different process corner and calibrated on its own",
             ha="center", color=MUT, fontsize=9)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(os.path.join(HERE, "figures", "alu4_levels.png"), dpi=160)
    plt.close(fig)


def layout():
    from PIL import Image, ImageDraw, ImageFont
    import klayout.db as db
    from render_chip import render
    gds = os.path.join(HERE, "klayout", "EBeam_heaters_atharvdubey_ALU4.gds")
    png = os.path.join(HERE, "figures", "alu4_layout.png")
    w, h = 2400, 2000
    render(gds, png, w, h, labels=False)
    ly = db.Layout()
    ly.read(gds)
    bb0 = ly.top_cell().dbbox()
    bb = db.DBox(bb0.left - 300, bb0.bottom - 20, bb0.right + 20, bb0.top + 20)
    s = min(w / bb.width(), h / bb.height())
    ox, oy = (w - bb.width() * s) / 2, (h - bb.height() * s) / 2

    def px(x, y):
        return ox + (x - bb.left) * s, oy + (bb.top - y) * s

    im = Image.open(png).convert("RGB")
    d = ImageDraw.Draw(im)
    try:
        f = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 40)
        g = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 24)
    except OSError:
        f = g = ImageFont.load_default()
    for bit, (gx, gy) in {0: (0, 0), 1: (1, 0), 2: (0, 1), 3: (1, 1)}.items():
        x0, y0 = gx * 1850.0, gy * 1524.0
        a, b = px(x0 + 60, y0 + 1520), px(x0 + 1845, y0 + 40)
        d.rounded_rectangle([a, b], radius=14, outline=(235, 163, 64), width=3)
        X, Y = px(x0 + 1520, y0 + 1400)
        d.text((X, Y), f"bit {bit}", fill=(235, 163, 64), font=f, anchor="mm")
        for yy, t in ((y0 + 940, "A, B half adder"), (y0 + 178, "carry-in half adder")):
            X, Y = px(x0 + 1250, yy)
            d.text((X, Y), t, fill=(139, 158, 165), font=g, anchor="mm")
    im.save(png)


if __name__ == "__main__":
    os.makedirs(os.path.join(HERE, "figures"), exist_ok=True)
    levels()
    layout()
    print("ok")
