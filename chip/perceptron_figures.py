#!/usr/bin/env python3
"""Figures for the optical neuron (needs klayout + matplotlib + Pillow; run after perceptron_export_web.py).

  figures/perceptron_layout.png      the drawn chip, labelled
  figures/perceptron_linearity.png   output against the ideal sum, and the worst error per corner
  figures/perceptron_digits.png      the stored digits and the chip's ten scores for two inputs
  figures/perceptron_eyes.png        output eyes for a stream of digit images
  figures/perceptron_rates.png       level opening against bit rate
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
RES = os.path.join(HERE, "perceptron_results.json")
SPEED = os.path.join(HERE, "speed", "perceptron_speed_results.json")
WEB = os.path.join(HERE, "..", "perceptron", "data.json")


def layout():
    from PIL import Image, ImageDraw, ImageFont
    import klayout.db as db
    sys.path.insert(0, HERE)
    from render_chip import render
    gds = os.path.join(HERE, "klayout", "EBeam_heaters_atharvdubey_Perceptron16.gds")
    png = os.path.join(OUT, "perceptron_layout.png")
    w = h = 2000
    render(gds, png, w=w, h=h, labels=False)
    ly = db.Layout()
    ly.read(gds)
    b0 = ly.top_cell().dbbox()
    bb = db.DBox(b0.left - 300, b0.bottom - 20, b0.right + 20, b0.top + 20)
    s = min(w / bb.width(), h / bb.height())
    ox, oy = (w - bb.width() * s) / 2, (h - bb.height() * s) / 2

    def px(x, y):                       # design coordinates (perceptron_siepic.py) -> pixels
        x, y = x + 120.0, y + 1040.0
        return ox + (x - bb.left) * s, oy + (bb.top - y) * s

    im = Image.open(png).convert("RGB")
    d = ImageDraw.Draw(im)
    try:
        f = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 24)
        fb = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf", 24)
    except OSError:
        f = fb = ImageFont.load_default()
    for y, t in ((127, "OUT +"), (0, "LASER in"), (-127, "OUT -")):
        X, Y = px(-30, y)
        d.text((X, Y), t, fill=(227, 235, 237), font=f, anchor="rm")
    lab = [((340, 72.75), "1 to 16\nsplit"), ((1520, 72.75), "16 to 1\ncombine"), ((1881, -45), "coup-\nler"),
           ((318, -755), "reference\narm"), ((185, -770), "trim"),
           ((940, 72.75), "16 arms, one phase section each")]
    for (x, y), t in lab:
        X, Y = px(x, y)
        d.multiline_text((X, Y), t, fill=(139, 158, 165), font=f, anchor="mm", align="center")
    im.save(png)


def linearity():
    r = json.load(open(RES))
    ex = r["example_nominal"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(12.5, 4.9), gridspec_kw=dict(width_ratios=[1.35, 1]))
    rng = np.random.default_rng(0)
    for S, pts in ex["points_uW_per_mW"].items():
        pts = np.array(pts)
        a.scatter(int(S) + rng.uniform(-0.35, 0.35, pts.size), pts, s=8, color=BLUE, alpha=0.6, lw=0)
    g = ex["gain_uW_per_level_at_1mW"]
    xs = np.array([-16, 16])
    a.plot(xs, xs / 2 * g, color=FAINT, lw=1, zorder=0)
    a.set_xticks(range(-16, 17, 4))
    a.set_xlabel("ideal sum S = sum of w x  (16 - 2 x arms that disagree)")
    a.set_ylabel("balanced output (uW, 1 mW laser)")
    a.set_title(f"one calibrated die, nominal corner: every point within {ex['max_error_levels']:.2f} of a level", fontsize=10, color=FG)
    names = list(r["corners"])
    worst = [r["corners"][k]["worst_max_error_levels"] for k in names]
    unc = [r["corners"][k]["worst_uncorrected_max_error_levels"] for k in names]
    lab = [k.replace("nm_x_", "/").replace("nm", "") for k in names]
    y = np.arange(len(names))
    b.barh(y - 0.18, unc, color=FAINT, height=0.34, label="phase calibration only")
    b.barh(y + 0.18, worst, color=BLUE, height=0.34, label="+ crosstalk map + balance weight")
    for i, v in enumerate(worst):
        b.text(v + 0.01, i + 0.18, f"{v:.2f}", va="center", color=FG, fontsize=8.5)
    b.axvline(0.5, color=ORANGE, lw=1.5, ls="--")
    b.text(0.51, -0.85, "levels merge beyond 0.5", color=ORANGE, fontsize=8.5, ha="left", va="center")
    b.set_yticks(y, lab)
    b.set_ylim(len(names) - 0.45, -1.3)
    b.set_xlim(0, 0.8)
    b.set_xlabel("worst error over 20 dies (levels)")
    b.set_ylabel("silicon thickness / waveguide width (nm)")
    b.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.45, -0.13), ncol=2, fontsize=8.5)
    b.set_title("all 17 levels apart on every die, once calibrated", fontsize=10, color=FG)
    b.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "perceptron_linearity.png"), dpi=150)
    plt.close(fig)


def chip_scores(D, img):
    """The ten outputs of the exported nominal die for one image (numpy port of the page's model)."""
    c = D["corners"]["220nm_x_500nm"]
    X = np.array(D["X"])
    tp = np.array([complex(*v) for v in c["t_plus"]]); tm = np.array([complex(*v) for v in c["t_minus"]])
    eta, phi0 = np.array(c["eta"]), np.array(c["phi0"])
    pp, pm = np.array(c["p_plus_mW"]), np.array(c["p_minus_mW"])
    x = np.ones(16, int)
    for i, a in enumerate(D["arm_of_pixel"]):
        x[a] = img[i]
    out = []
    for d in range(10):
        w = np.ones(16, int)
        for i, ch in enumerate(D["font"][str(d)]):
            w[D["arm_of_pixel"][i]] = 1 if ch == "1" else -1
        s = x * w
        pa = np.where(s > 0, pp[:16], pm[:16])
        err = np.array(c["M"]) @ (pa - pp[:16])                    # crosstalk the pattern sends each arm (rad)
        pa = np.clip(pa - np.array(c["K"]) @ err, 0, 80)           # pre-corrected heater powers (mW)
        p = np.concatenate([pa, pp[16:]])
        ph = (p * eta) @ X.T + phi0
        e = np.exp(1j * ph)
        out.append((abs(e @ tp) ** 2 - c["lam"] * abs(e @ tm) ** 2) * 1e3)   # uW at 1 mW
    return np.array(out)


def digits():
    D = json.load(open(WEB))
    font = D["font"]
    fig = plt.figure(figsize=(12.5, 5.2))
    gs = fig.add_gridspec(2, 12, height_ratios=[1, 1.5], hspace=0.45, wspace=0.6)
    for d in range(10):
        ax = fig.add_subplot(gs[0, d + 1 if d < 10 else d])
        g = np.array([int(ch) for ch in font[str(d)]]).reshape(5, 3)
        ax.imshow(g, cmap=matplotlib.colors.ListedColormap(["#16232a", "#ff5a47"]), vmin=0, vmax=1)
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        ax.set_title(str(d), color=FG, fontsize=11)
        for sp_ in ax.spines.values():
            sp_.set_color("#33464f")
    fig.text(0.5, 0.965, "stored weight patterns: lit pixel = +1, dark = -1 (the 16th arm is a fixed +1 bias)", ha="center", color=MUT, fontsize=9.5)
    cases = (("a 7 with one pixel flipped", "111001011001001"), ("an 8: one pixel from a 0, a 6 and a 9", "111101111101111"))
    for j, (title, s) in enumerate(cases):
        img = np.array([1 if ch == "1" else -1 for ch in s])
        sc = chip_scores(D, img)
        ideal = []
        for d in range(10):
            w = np.array([1 if ch == "1" else -1 for ch in font[str(d)]])
            ideal.append(int((img * w).sum()) + 1)
        ax = fig.add_subplot(gs[1, 1 + 6 * j: 6 + 6 * j])
        lvl = sc.max() - sc.min()
        cols = [ORANGE if k == int(np.argmax(sc)) else BLUE for k in range(10)]
        ax.bar(range(10), sc, color=cols, width=0.65)
        for k in range(10):
            ax.text(k, sc[k] + 0.02 * lvl * (1 if sc[k] >= 0 else -1), str(ideal[k]), ha="center", va="bottom" if sc[k] >= 0 else "top", color=MUT, fontsize=8.5)
        ax.axhline(0, color=FAINT, lw=1)
        ax.set_ylim(min(sc.min(), 0) * 1.5 - 0.08 * lvl, sc.max() * 1.15)
        ax.set_xticks(range(10)); ax.grid(axis="x", visible=False)
        ax.set_xlabel("stored digit (number above bar: ideal sum)")
        if j == 0:
            ax.set_ylabel("output (uW, 1 mW laser)")
        ax.set_title(f"input: {title} -> chip says {int(np.argmax(sc))}", fontsize=10, color=FG)
        ins = ax.inset_axes([1.02, 0.55, 0.11, 0.42])
        ins.imshow(np.array([int(ch) for ch in s]).reshape(5, 3), cmap=matplotlib.colors.ListedColormap(["#16232a", "#8fe3ff"]))
        ins.set_xticks([]); ins.set_yticks([]); ins.grid(False)
    fig.savefig(os.path.join(OUT, "perceptron_digits.png"), dpi=150, bbox_inches="tight", facecolor=INK)
    plt.close(fig)


def eyes():
    z = np.load(os.path.join(HERE, "speed", "perceptron_eyes.npz"))
    runs = json.load(open(SPEED))["runs"]
    rows = (("ideal", "aligned", "aligned drives\nideal modulators", BLUE),
            ("lithium_niobate_100GHz", "aligned", "aligned drives\n100 GHz modulators", BLUE),
            ("lithium_niobate_100GHz", "unaligned", "unaligned drives\n100 GHz modulators", ORANGE))
    rates = (25, 50, 100, 200)
    full = max(float(np.abs(z[k]).max()) for k in z.files) * 1.05
    fig, axs = plt.subplots(3, 4, figsize=(12.5, 7.8), sharex=True)
    for r_, (io, mode, label, col) in enumerate(rows):
        for c_, rate in enumerate(rates):
            ax = axs[r_, c_]
            f = z[f"{io}__{mode}__{rate}"]
            x = np.linspace(-1, 1, f.shape[1])
            for row in f[:160]:
                ax.plot(x, row / 1e3, color=col, lw=0.6, alpha=0.25)
            run = next(q for q in runs if q["io"] == io and q["mode"] == mode and int(q["rate_Gbps"]) == rate)
            ok = run["level_opening"] > 0 and run["level_width"] >= 0.3
            ax.set_title(f"{rate} Gb/s: " + ("all levels apart" if ok else "levels merge"), fontsize=9.5, color=FG if ok else ORANGE)
            ax.set_ylim(-full / 1e3, full / 1e3)
            ax.grid(axis="x", visible=False)
            if c_ == 0:
                ax.set_ylabel(label + "\n\nbalanced current (mA)", fontsize=9)
            if r_ == 2:
                ax.set_xlabel("time (bit periods)")
    fig.text(0.5, 0.005, "Balanced output for a stream of 3x5 digit images, one image per bit slot, weights = the stored 7; 10 mW laser",
             ha="center", color=MUT, fontsize=9)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(os.path.join(OUT, "perceptron_eyes.png"), dpi=150)
    plt.close(fig)


def rates():
    runs = json.load(open(SPEED))["runs"]
    series = (("lithium_niobate_100GHz", "aligned", "-", "o", BLUE, "100 GHz I/O"),
              ("silicon_50GHz", "aligned", "-", "o", AQUA, "50 GHz I/O"),
              ("lithium_niobate_100GHz", "unaligned", ":", "s", ORANGE, "unaligned, 100 GHz"))
    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    for io, mode, ls, mk, col, lab in series:
        rs = sorted((q for q in runs if q["io"] == io and q["mode"] == mode), key=lambda q: q["rate_Gbps"])
        x = [q["rate_Gbps"] for q in rs]
        y = [max(0.0, 100 * q["level_opening"]) for q in rs]
        ax.plot(x, y, ls, color=col, lw=2, marker=mk, ms=7, mfc=col, mec=INK, mew=2)
        pos = [i for i, v in enumerate(y) if v > 0]
        i = max(pos) if pos else 0
        if mode == "unaligned":
            ax.text(36, 30, lab, color=ORANGE, fontsize=9, ha="left")
        else:
            ax.annotate(lab, (x[i], y[i]), xytext=(8, 6), textcoords="offset points", color=FG, fontsize=9)
    ax.set_xlabel("images per second (G) = bit rate per input (Gb/s)")
    ax.set_ylabel("smallest gap between neighbouring levels (% of a level)")
    ax.set_ylim(0, 105)
    ax.set_xlim(20, 210)
    ax.set_title("Sixteen inputs summed in one pass: the modulators set the rate", fontsize=10, color=FG)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "perceptron_rates.png"), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    which = sys.argv[1:] or ["layout", "linearity", "digits", "eyes", "rates"]
    for w in which:
        globals()[w]()
    print("ok", which)
