#!/usr/bin/env python3
"""Figures for the mesh chip (needs klayout + matplotlib + Pillow; run after mesh_export_web.py).

  figures/mesh_layout.png    the drawn chip, labelled
  figures/mesh_corners.png   test accuracy of every simulated die at the nine corners, against software
  figures/mesh_hidden.png    the chip's layer-1 outputs against the software network's (example die)
  figures/mesh_speed.png     accuracy and layer-1 error against symbol rate
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
BLUE, ORANGE, AQUA, RED = "#3987e5", "#d95926", "#199e70", "#ff5a47"
plt.rcParams.update({"figure.facecolor": INK, "axes.facecolor": INK, "axes.edgecolor": "#33464f", "axes.labelcolor": MUT,
                     "xtick.color": MUT, "ytick.color": MUT, "text.color": FG, "font.size": 10,
                     "axes.grid": True, "grid.color": "#1c2a31"})
RES = os.path.join(HERE, "mesh_results.json")
SPEED = os.path.join(HERE, "speed", "mesh_speed_results.json")


def layout():
    from PIL import Image, ImageDraw, ImageFont
    import klayout.db as db
    sys.path.insert(0, HERE)
    from render_chip import render
    gds = os.path.join(HERE, "klayout", "EBeam_heaters_atharvdubey_Mesh8.gds")
    png = os.path.join(OUT, "mesh_layout.png")
    w, h = 3200, 1150
    render(gds, png, w=w, h=h, labels=False)
    ly = db.Layout()
    ly.read(gds)
    b0 = ly.top_cell().dbbox()
    bb = db.DBox(b0.left - 300, b0.bottom - 20, b0.right + 20, b0.top + 20)
    s = min(w / bb.width(), h / bb.height())
    ox, oy = (w - bb.width() * s) / 2, (h - bb.height() * s) / 2
    rep = json.load(open(os.path.join(HERE, "klayout", "mesh_build_report.json")))
    # design -> GDS shift used by mesh_siepic.py (pads at the bottom start 100 um above the die edge)
    comps = json.load(open(os.path.join(HERE, "klayout", "mesh_extracted_netlist.json")))["components"]
    gc = {c["role"]: c["center"] for c in comps if c["kind"].startswith("GC_")}
    heat = {c["role"]: c["center"] for c in comps if c["kind"] == "wg_heater"}

    def px(x, y):                       # GDS coordinates -> pixels
        return ox + (x - bb.left) * s, oy + (bb.top - y) * s

    im = Image.open(png).convert("RGB")
    d = ImageDraw.Draw(im)
    try:
        f = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 21)
    except OSError:
        f = ImageFont.load_default()
    for role, t in (("laser", "LASER in"), ("o7", "OUT 7"), ("o0", "OUT 0"), ("o1", "OUT 1")):
        X, Y = px(gc[role][0] - 45, gc[role][1])
        d.text((X, Y), t, fill=(227, 235, 237), font=f, anchor="rm")
    ymid = gc["laser"][1]
    xs_enc = np.mean([heat[f"E{i}A"][0] for i in range(8)])
    x_mesh0 = min(v[0] for k, v in heat.items() if k.startswith(("T", "F")))
    x_mesh1 = max(v[0] for k, v in heat.items() if k.startswith(("T", "F")))
    lab = [((xs_enc, ymid), "8 encoders"), ((x_mesh0 - 230, ymid), "converger"),
           ((0.5 * (x_mesh0 + x_mesh1), ymid - 520), "8 x 8 mesh: 28 interferometers, 56 heaters")]
    for (x, y), t in lab:
        X, Y = px(x, y)
        box = d.textbbox((X, Y), t, font=f, anchor="mm")
        d.rectangle((box[0] - 8, box[1] - 5, box[2] + 8, box[3] + 5), fill=(10, 16, 19))
        d.text((X, Y), t, fill=(200, 212, 216), font=f, anchor="mm")
    im.save(png)


def corners():
    res = json.load(open(RES))
    sw = res["network"]["software_accuracy"]
    fig, ax = plt.subplots(figsize=(9.5, 4.2))
    labels = []
    for k, c in enumerate(res["corners"]):
        accs = [d["accuracy"] * 100 for d in res["dies"] if d["corner"] == c["corner"]]
        ax.scatter(np.full(len(accs), k) + np.linspace(-0.18, 0.18, len(accs)), accs, s=16, color=BLUE, zorder=3)
        ax.plot([k - 0.28, k + 0.28], [c["accuracy_mean"] * 100] * 2, color=FG, lw=1.5, zorder=4)
        labels.append(f"{int(c['corner'][0])}/{int(c['corner'][1])}")
    ax.axhline(sw * 100, color=RED, lw=1.4)
    ax.text(len(labels) - 0.5, sw * 100 + 0.08, f"same network in software {sw * 100:.2f}%", color=RED, ha="right", va="bottom", fontsize=9)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_xlabel("process corner: silicon thickness / waveguide width (nm)")
    ax.set_ylabel("test accuracy, 10,000 digits (%)")
    lo = min(d["accuracy"] for d in res["dies"]) * 100
    ax.set_ylim(np.floor(lo - 0.5), sw * 100 + 0.8)
    ax.set_title(f"{len(res['dies'])} simulated dies, each calibrated and programmed from its three outputs", color=FG, fontsize=11, loc="left")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "mesh_corners.png"), dpi=160)
    plt.close(fig)


def hidden():
    res = json.load(open(RES))
    d = res["example"]["detail"]
    hc, hr = np.array(d["h_chip"]), np.array(d["h_ref"])
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    cols = plt.cm.plasma(np.linspace(0.35, 0.95, hc.shape[1]))
    for k in range(hc.shape[1]):
        ax.scatter(hr[:, k], hc[:, k], s=5, color=cols[k], alpha=0.7, label=f"h{k + 1}")
    m = max(hc.max(), hr.max()) * 1.03
    ax.plot([0, m], [0, m], color=FAINT, lw=1)
    ax.set_xlim(0, m)
    ax.set_ylim(0, m)
    ax.set_xlabel("software network: |t . x|^2")
    ax.set_ylabel("simulated chip, measured and scaled")
    ax.set_title(f"layer 1, 400 test digits, nominal die\nRMS difference {res['example']['hidden_rms_error'] * 100:.1f}% of the outputs' RMS",
                 color=FG, fontsize=10, loc="left")
    ax.legend(fontsize=7, ncol=2, frameon=False, markerscale=2)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "mesh_hidden.png"), dpi=160)
    plt.close(fig)


def speed():
    sp = json.load(open(SPEED))
    fig, axs = plt.subplots(1, 2, figsize=(10.5, 4.1))
    cols = {"ideal": FG, "lithium_niobate_100GHz": BLUE, "silicon_50GHz": ORANGE}
    names = {"ideal": "ideal modulators", "lithium_niobate_100GHz": "100 GHz", "silicon_50GHz": "50 GHz"}
    dc = sp["dc"]["accuracy"] * 100
    for io in ("ideal", "lithium_niobate_100GHz", "silicon_50GHz"):
        rs = sorted([r for r in sp["runs"] if r["io"] == io and r["aligned"]], key=lambda r: r["rate_GBd"])
        x = [r["rate_GBd"] for r in rs]
        axs[0].plot(x, [r["accuracy_noiseless"] * 100 for r in rs], "o-", color=cols[io], label=names[io], ms=4)
    rs = sorted([r for r in sp["runs"] if r["io"] == "lithium_niobate_100GHz" and r["aligned"]], key=lambda r: r["rate_GBd"])
    x = [r["rate_GBd"] for r in rs]
    for key, ls, nm in (("noiseless", "-", "no noise"), ("100mW", "--", "100 mW laser"), ("10mW", ":", "10 mW laser")):
        axs[1].plot(x, [r[f"accuracy_{key}"] * 100 for r in rs], "o" + ls, color=BLUE, label=nm, ms=4)
    for a in axs:
        a.axhline(dc, color=RED, lw=1.2)
        a.set_xscale("log")
        a.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        a.text(390, dc + 0.8, "same chip, slow", color=RED, fontsize=9, ha="right")
        a.set_xticks([25, 50, 100, 200, 400])
        a.set_xticklabels(["25", "50", "100", "200", "400"])
        a.set_xlabel("digits per second through each page (billions)")
        a.set_ylim(min(30, a.get_ylim()[0]), dc + 4)
        a.legend(frameon=False, fontsize=9, loc="lower left")
    axs[0].set_ylabel(f"accuracy on {sp['n_digits']} test digits (%)")
    axs[0].set_title("what the bandwidth costs (no noise)", color=FG, fontsize=10, loc="left")
    axs[1].set_title("with photodiode noise, 100 GHz modulators", color=FG, fontsize=10, loc="left")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "mesh_speed.png"), dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    for f in sys.argv[1:] or ("layout", "corners", "hidden", "speed"):
        globals()[f]()
        print("done", f)
