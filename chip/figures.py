#!/usr/bin/env python3
"""Figures from full_adder_results.json."""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
INK, MUT, FG = "#0a1013", "#8b9ea5", "#e3ebed"
plt.rcParams.update({"figure.facecolor": INK, "axes.facecolor": INK, "axes.edgecolor": "#33464f", "axes.labelcolor": MUT,
                     "xtick.color": MUT, "ytick.color": MUT, "text.color": FG, "font.size": 10,
                     "axes.grid": True, "grid.color": "#1c2a31"})


def main():
    r = json.load(open(os.path.join(HERE, "full_adder_results.json")))
    os.makedirs(os.path.join(HERE, "figures"), exist_ok=True)
    m = r["majority"]
    rows = m["truth_table"]
    labels = [f"{t['A']}{t['B']}{t['Cin']}" for t in rows]
    fig, axs = plt.subplots(1, 3, figsize=(11, 3.4))
    for ax, (key, name, truth) in zip(axs, (("xor1_uA", "XOR 1 (A xor B)", lambda t: t["A"] ^ t["B"]),
                                            ("sum_uA", "SUM", lambda t: t["A"] ^ t["B"] ^ t["Cin"]),
                                            ("carry_uA", "CARRY", lambda t: (t["A"] & t["B"]) | (t["Cin"] & (t["A"] ^ t["B"]))))):
        v = [t[key] for t in rows]
        cols = ["#2c9fd6" if truth(t) else "#5d6f77" for t in rows]
        ax.bar(range(8), v, color=cols)
        thr = m["margins"][key.split("_")[0]]["threshold_uA"]
        ax.axhline(thr, color="#eba340", lw=1)
        ax.set_xticks(range(8), labels)
        ax.set_title(f"{name}: margin {m['margins'][key.split('_')[0]]['margin_pct']:.0f}%", color=FG, fontsize=10)
        ax.set_xlabel("inputs A B Cin")
    axs[0].set_ylabel("photocurrent (uA), 1 mW laser")
    fig.text(0.5, 0.005, "blue: output should be 1   grey: should be 0   amber: comparator threshold", ha="center", color=MUT, fontsize=9)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(os.path.join(HERE, "figures", "truth_table.png"), dpi=160)
    plt.close(fig)

    c = r["cascade_failure"]
    fig, ax = plt.subplots(figsize=(6.4, 3.5))
    ax.plot(c["phase_deg"], c["A_only_plus_Cin"], color="#e5503f", label="A=1, B=0, Cin=1 (should be 0)")
    ax.plot(c["phase_deg"], c["B_only_plus_Cin"], color="#2c9fd6", label="A=0, B=1, Cin=1 (should be 0)")
    ax.axhline(0.5, color="#eba340", lw=1, label="threshold (half the '1' level)")
    ax.set_xlabel("carry-in beam phase (degrees)")
    ax.set_ylabel("sum output / '1' level")
    ax.set_title("Light-only cascade (Fig. 15 as drawn): no phase makes both dark", color=FG, fontsize=10)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "figures", "cascade_failure.png"), dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
