#!/usr/bin/env python3
"""Figures for the phase-encoded parity chip (needs klayout + matplotlib + Pillow).

  figures/parity_layout.png        the drawn chip, labelled
  figures/parity_levels.png        every input's output level after calibration, and on/off per corner
  figures/parity_accumulation.png  how drive errors add up along the chain, against the amplitude cascade
  figures/parity_eyes.png          eye diagrams of 8-input parity, aligned and unaligned drives
  figures/parity_rates.png         eye opening against bit rate
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "figures")
INK, MUT, FG, FAINT = "#0a1013", "#8b9ea5", "#e3ebed", "#5d6f77"
BLUE, ORANGE, AQUA = "#3987e5", "#d95926", "#199e70"
plt.rcParams.update({"figure.facecolor": INK, "axes.facecolor": INK, "axes.edgecolor": "#33464f", "axes.labelcolor": MUT,
                     "xtick.color": MUT, "ytick.color": MUT, "text.color": FG, "font.size": 10,
                     "axes.grid": True, "grid.color": "#1c2a31"})
RES = os.path.join(HERE, "parity_results.json")
SPEED = os.path.join(HERE, "speed", "parity_speed_results.json")


def layout():
    from PIL import Image, ImageDraw, ImageFont
    import klayout.db as db
    sys.path.insert(0, HERE)
    from render_chip import render
    gds = os.path.join(HERE, "klayout", "EBeam_heaters_atharvdubey_Parity8.gds")
    png = os.path.join(OUT, "parity_layout.png")
    w, h = 2400, 860
    render(gds, png, w=w, h=h, labels=False)
    ly = db.Layout()
    ly.read(gds)
    b0 = ly.top_cell().dbbox()
    bb = db.DBox(b0.left - 300, b0.bottom - 20, b0.right + 20, b0.top + 20)
    s = min(w / bb.width(), h / bb.height())
    ox, oy = (w - bb.width() * s) / 2, (h - bb.height() * s) / 2

    def px(x, y):                       # design coordinates (parity_siepic.py) -> image pixels
        x, y = x + 120.0, y + 130.0
        return ox + (x - bb.left) * s, oy + (bb.top - y) * s

    im = Image.open(png).convert("RGB")
    d = ImageDraw.Draw(im)
    try:
        f = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 22)
        fb = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf", 22)
    except OSError:
        f = fb = ImageFont.load_default()
    for y, t in ((254, "PARITY"), (127, "LASER in"), (0, "NOT PARITY")):
        X, Y = px(-30, y)
        d.text((X, Y), t, fill=(227, 235, 237), font=f, anchor="rm")
    for k in range(9):
        X, Y = px(160 + 150 * k + 60, 172)
        d.text((X, Y), f"S{k + 1}" if k < 8 else "bias", fill=(235, 163, 64), font=fb, anchor="mm")
    X, Y = px(880, 70)
    d.text((X, Y), "reference arm: mirror image, plain waveguide", fill=(139, 158, 165), font=f, anchor="mm")
    X, Y = px(880, 140)
    d.text((X, Y), "signal arm: one phase section per input bit", fill=(139, 158, 165), font=f, anchor="mm")
    X, Y = px(1600, 160)
    d.text((X, Y), "coupler", fill=(139, 158, 165), font=f, anchor="mm")
    im.save(png)


def levels():
    r = json.load(open(RES))["truth"]
    nom = r["220nm_x_500nm"]["example"]["levels_uW"]
    idx = np.arange(256)
    ones = np.array([bin(i).count("1") for i in idx])
    par = ones % 2
    fig, (a, b) = plt.subplots(1, 2, figsize=(12.5, 4.8), gridspec_kw=dict(width_ratios=[1.35, 1]))
    rng = np.random.default_rng(0)
    p = np.array(nom["parity"])
    for want, col, fill in ((1, BLUE, True), (0, ORANGE, False)):
        m = par == want
        x = ones[m] + rng.uniform(-0.18, 0.18, m.sum())
        a.scatter(x, p[m], s=16, facecolors=col if fill else "none", edgecolors=col, linewidths=1.0, zorder=3)
    a.set_yscale("log")
    a.set_xticks(range(9))
    a.set_xlabel("number of 1s among the eight inputs")
    a.set_ylabel("PARITY output (uW, 1 mW laser)")
    a.set_title("all 256 inputs, one calibrated die at the nominal corner", fontsize=10, color=FG)
    a.grid(axis="x", visible=False)
    a.text(0.02, 0.5, "filled blue: odd number of 1s (should be bright)\nhollow red: even (should be dark)",
           transform=a.transAxes, color=MUT, fontsize=9, va="center")
    names = list(r)
    worst = [r[k]["worst_on_off_dB"] for k in names]
    lab = [k.replace("nm_x_", "/").replace("nm", "") for k in names]
    b.barh(range(len(names)), worst, color=BLUE, height=0.62)
    for i, v in enumerate(worst):
        b.text(v + 0.6, i, f"{v:.0f} dB", va="center", color=FG, fontsize=9)
    b.set_yticks(range(len(names)), lab)
    b.invert_yaxis()
    b.set_xlim(0, 50)
    b.set_xlabel("worst on/off over 20 dies x 256 inputs (dB)")
    b.set_ylabel("silicon thickness / waveguide width (nm)")
    b.set_title("every input correct on all 180 dies", fontsize=10, color=FG)
    b.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "parity_levels.png"), dpi=150)
    plt.close(fig)


def accumulation():
    res = json.load(open(RES))
    a = res["accumulation"]
    n = np.array(a["sections"])
    m = n <= 16
    fig, ax = plt.subplots(figsize=(8.6, 5.0))
    for key, col, ls, lab in (("random_2pct", BLUE, "-", "2% random drive error per section"),
                              ("random_5pct", ORANGE, "-", "5% random"),
                              ("systematic_2pct", AQUA, "--", "2% systematic (all sections the same)")):
        y = np.array(a[key])[m]
        ax.plot(n[m], y, ls, color=col, lw=2, marker="o", ms=5, mfc=col, mec=INK, mew=1.5, label=lab)
    nom = res["truth"]["220nm_x_500nm"]["worst_on_off_dB"]
    ax.scatter([8], [nom], s=90, marker="D", color=FG, zorder=5)
    ax.annotate(f"this chip, calibrated: {nom:.0f} dB", (8, nom), xytext=(10, -4), textcoords="offset points", color=FG, fontsize=9)
    amp = 10 * np.log10(1 / res["cascade"]["amplitude_encoded"]["best_worst_case_dark_level"])
    ax.scatter([2], [amp], s=90, marker="X", color=ORANGE, zorder=5)
    ax.annotate("amplitude-encoded XORs (the full adder):\nfails at the second stage", (2, amp), xytext=(14, -6), textcoords="offset points",
                color=ORANGE, fontsize=9, va="top")
    ax.axhline(0, color=FAINT, lw=1)
    ax.set_xlim(0.5, 16.5)
    ax.set_ylim(-12, 48)
    ax.set_xlabel("sections in the chain (input bits)")
    ax.set_ylabel("on/off of the worst input (dB, 95% of dies)")
    ax.set_title("Phases add, so logic cascades; their errors add too, so nothing restores the signal", fontsize=10, color=FG)
    ax.legend(frameon=False, loc="upper right", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "parity_accumulation.png"), dpi=150)
    plt.close(fig)


def eyes():
    z = np.load(os.path.join(HERE, "speed", "parity_eyes.npz"))
    runs = json.load(open(SPEED))["runs"]
    rows = (("ideal", "aligned", "aligned drives\nideal modulators", BLUE),
            ("lithium_niobate_100GHz", "aligned", "aligned drives\n100 GHz modulators", BLUE),
            ("lithium_niobate_100GHz", "unaligned", "unaligned drives\n100 GHz modulators", ORANGE))
    rates = (25, 50, 100, 200)
    fig, axs = plt.subplots(3, 4, figsize=(12.5, 7.6), sharex=True)
    for r_, (io, mode, label, col) in enumerate(rows):
        for c_, rate in enumerate(rates):
            ax = axs[r_, c_]
            f = z[f"{io}__{mode}__{rate}"]
            x = np.linspace(-1, 1, f.shape[1])
            for row in f[:160]:
                ax.plot(x, row, color=col, lw=0.6, alpha=0.25)
            run = next(q for q in runs if q["io"] == io and q["mode"] == mode and int(q["rate_Gbps"]) == rate)
            ok = run["Q"] >= 7 and run["correct"] and run["eye_width_pct"] >= 30
            ax.set_title(f"{rate} Gb/s ({1000 / rate:.0f} ps bit): " + ("open" if ok else "closed"), fontsize=9.5, color=FG if ok else ORANGE)
            ax.set_ylim(-60, 2600)
            ax.grid(axis="x", visible=False)
            if c_ == 0:
                ax.set_ylabel(label + "\n\nphotocurrent (uA)", fontsize=9)
            if r_ == 2:
                ax.set_xlabel("time (bit periods)")
    fig.text(0.5, 0.005, "PARITY output for eight independent random bit streams, read straight off the chip; 10 mW laser, 1 A/W photodiode",
             ha="center", color=MUT, fontsize=9)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(os.path.join(OUT, "parity_eyes.png"), dpi=150)
    plt.close(fig)


def rates():
    runs = json.load(open(SPEED))["runs"]
    series = (("lithium_niobate_100GHz", "aligned", "-", "o", BLUE, "8-input parity, 100 GHz I/O"),
              ("lithium_niobate_100GHz", "wire", "--", None, BLUE, None),
              ("silicon_50GHz", "aligned", "-", "o", AQUA, "8-input parity, 50 GHz I/O"),
              ("silicon_50GHz", "wire", "--", None, AQUA, None),
              ("lithium_niobate_100GHz", "unaligned", ":", "s", ORANGE, "unaligned drives, 100 GHz I/O"))
    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    for io, mode, ls, mk, col, lab in series:
        rs = sorted((q for q in runs if q["io"] == io and q["mode"] == mode), key=lambda q: q["rate_Gbps"])
        x = [q["rate_Gbps"] for q in rs]
        y = [max(0.0, q["eye_opening_pct"]) for q in rs]
        ax.plot(x, y, ls, color=col, lw=2, marker=mk, ms=7, mfc=col, mec=INK, mew=2)
        if lab:
            pos = [i for i, v in enumerate(y) if v > 0]
            i = max(pos) if pos else 0
            if mode == "unaligned":
                ax.text(59, 30, "unaligned drives,\n100 GHz I/O", color=ORANGE, fontsize=9, ha="right", va="center")
            else:
                ax.annotate(lab, (x[i], y[i]), xytext=(8, 6), textcoords="offset points", color=FG, fontsize=9)
    ax.set_xlabel("bit rate (Gb/s)")
    ax.set_ylabel("eye opening (% of the '1' level)")
    ax.set_ylim(0, 105)
    ax.set_xlim(20, 250)
    ax.plot([], [], "-", color=MUT, lw=2, marker="o", ms=6, label="parity of eight streams")
    ax.plot([], [], "--", color=MUT, lw=2, label="same path, one stream, no logic")
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2, fontsize=9)
    ax.set_title("Aligned drives: parity tracks a plain wire until the modulators' slow edges start to add up", fontsize=10, color=FG)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "parity_rates.png"), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    which = sys.argv[1:] or ["layout", "levels", "accumulation", "eyes", "rates"]
    for w in which:
        globals()[w]()
    print("ok", which)
