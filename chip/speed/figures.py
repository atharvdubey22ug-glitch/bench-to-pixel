#!/usr/bin/env python3
"""Figures for the speed study: eye diagrams and eye opening against bit rate."""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "figures")
INK, MUT, FG = "#0a1013", "#8b9ea5", "#e3ebed"
BLUE, ORANGE, AQUA = "#3987e5", "#d95926", "#199e70"          # validated dark categorical slots 1-3
plt.rcParams.update({"figure.facecolor": INK, "axes.facecolor": INK, "axes.edgecolor": "#33464f", "axes.labelcolor": MUT,
                     "xtick.color": MUT, "ytick.color": MUT, "text.color": FG, "font.size": 10,
                     "axes.grid": True, "grid.color": "#1c2a31"})


def eyes():
    z = np.load(os.path.join(HERE, "speed_eyes.npz"))
    res = json.load(open(os.path.join(HERE, "speed_results.json")))["runs"]
    rows = (("michelson", "ideal", "Michelson XOR as drawn\nideal modulators", AQUA),
            ("single_pass", "ideal", "single-pass XOR\nideal modulators", BLUE),
            ("single_pass", "lithium_niobate_100GHz", "single-pass XOR\n100 GHz modulators + receiver", BLUE))
    rates = (25, 50, 100, 200)
    fig, axs = plt.subplots(3, 4, figsize=(12.5, 7.6), sharex=True)
    for r_, (circ, io, label, col) in enumerate(rows):
        for c_, rate in enumerate(rates):
            ax = axs[r_, c_]
            k = f"{circ}__{io}__xor__{rate}"
            f = z[k]
            x = np.linspace(-1, 1, f.shape[1])
            for row in f[:160]:
                ax.plot(x, row, color=col, lw=0.6, alpha=0.25)
            run = next(r for r in res if r["circuit"] == circ and r["io"] == io and r["mode"] == "xor" and int(r["rate_Gbps"]) == rate)
            ok = run["Q"] >= 7 and run["correct"] and run["eye_width_pct"] >= 30
            ax.set_title(f"{rate} Gb/s ({1000 / rate:.0f} ps bit): " + ("open" if ok else "closed"), fontsize=9.5,
                         color=FG if ok else ORANGE)
            ax.set_ylim(-20, 700 if circ == "single_pass" else 175)
            ax.grid(axis="x", visible=False)
            if c_ == 0:
                ax.set_ylabel(label + "\n\nphotocurrent (uA)", fontsize=9)
            if r_ == 2:
                ax.set_xlabel("time (bit periods)")
    fig.text(0.5, 0.005, "XOR output of two independent 2^7-1 bit streams, read straight off the chip; 10 mW laser, 1 A/W photodiode",
             ha="center", color=MUT, fontsize=9)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(os.path.join(OUT, "speed_eyes.png"), dpi=150)
    plt.close(fig)


def rates():
    res = json.load(open(os.path.join(HERE, "speed_results.json")))["runs"]
    series = (("single_pass", "lithium_niobate_100GHz", "single-pass, 100 GHz I/O", BLUE),
              ("single_pass", "silicon_50GHz", "single-pass, 50 GHz I/O", ORANGE),
              ("michelson", "silicon_50GHz", "Michelson as drawn, 50 GHz I/O", AQUA))
    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    for circ, io, label, col in series:
        for mode, ls, mk in (("xor", "-", "o"), ("wire", "--", None)):
            rs = sorted((r for r in res if r["circuit"] == circ and r["io"] == io and r["mode"] == mode), key=lambda r: r["rate_Gbps"])
            x = [r["rate_Gbps"] for r in rs]
            y = [max(0.0, r["eye_opening_pct"]) for r in rs]
            ax.plot(x, y, ls, color=col, lw=2, marker=mk, ms=7, mfc=col, mec=INK, mew=2)
            if mode == "xor":
                if circ == "michelson":
                    ax.annotate(label, (37.5, 45.0), xytext=(52, 45), textcoords="data", color=FG, fontsize=9, va="center",
                                arrowprops=dict(arrowstyle="-", color=MUT, lw=0.8))
                else:
                    last = max(i for i, v in enumerate(y) if v > 0)
                    ax.annotate(label, (x[last], y[last]), xytext=(8, 0), textcoords="offset points", color=FG, fontsize=9, va="center")
    ax.set_xlabel("bit rate (Gb/s)")
    ax.set_ylabel("eye opening (% of the '1' level)")
    ax.set_ylim(0, 105)
    ax.set_xlim(20, 250)
    ax.plot([], [], "-", color=MUT, lw=2, marker="o", ms=6, label="XOR of two streams")
    ax.plot([], [], "--", color=MUT, lw=2, label="same path, no logic (one stream)")
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2, fontsize=9)
    ax.set_title("XOR (solid) against the same path with no logic (dashed): the modulators set the limit", fontsize=10, color=FG)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "speed_rates.png"), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    eyes()
    rates()
    print("ok")
