#!/usr/bin/env python3
"""Export the parity chip's simulation results for the web page (../parity/data.json, ../parity/layout.jpg)."""
import base64
import json
import os

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "parity")
TRACES, POINTS, FULL = 40, 100, 2600.0          # eye traces per case, points per trace, uA full scale


def main():
    os.makedirs(OUT, exist_ok=True)
    r = json.load(open(os.path.join(HERE, "parity_results.json")))
    sp = json.load(open(os.path.join(HERE, "speed", "parity_speed_results.json")))
    rep = json.load(open(os.path.join(HERE, "klayout", "parity_build_report.json")))
    corners = {}
    for k, v in r["truth"].items():
        lv = v["example"]["levels_uW"]
        corners[k] = dict(parity=[float(f"{x:.4g}") for x in lv["parity"]], nparity=[float(f"{x:.4g}") for x in lv["nparity"]],
                          worst_on_off_dB=v["worst_on_off_dB"], dies=v["dies"], all_correct=v["all_correct"],
                          calibration=v["example"]["calibration"])
    z = np.load(os.path.join(HERE, "speed", "parity_eyes.npz"))
    eyes = {}
    for run in sp["runs"]:
        key = f"{run['io']}|{run['mode']}|{int(run['rate_Gbps'])}"
        f = z[key.replace("|", "__")][:TRACES]
        f = f[:, np.linspace(0, f.shape[1] - 1, POINTS).round().astype(int)]
        q = np.clip(np.round(f / FULL * 255), 0, 255).astype(np.uint8)
        eyes[key] = dict(b64=base64.b64encode(q.tobytes()).decode(), n=int(q.shape[0]),
                         open_pct=round(float(run["eye_opening_pct"]), 1), Q=round(float(run["Q"]), 1),
                         width_pct=round(float(run["eye_width_pct"]), 1),
                         ok=bool(run["Q"] >= 7 and run["correct"] and run["eye_width_pct"] >= 30))
    data = dict(
        corners=corners,
        cascade=r["cascade"],
        accumulation={k: r["accumulation"][k] for k in ("sections", "random_2pct", "random_5pct", "systematic_2pct")},
        spectral={k: v for k, v in r["spectral"].items() if k.startswith("+-")},
        speed=dict(summary=sp["summary"], skew_ps=sp["section_skew_ps"],
                   latency_first_ps=sp["latency"]["S1"]["first_ps"], latency_last_ps=sp["latency"]["S8"]["first_ps"],
                   coupler_to_fibre_ps=sp["latency"]["coupler_to_fibre_ps"], eyes=eyes, points=POINTS, full_uA=FULL),
        chip=dict(die_um=rep["die_um"], components=rep["components"], errors=rep["siepic_verification_errors"],
                  arm_um=rep["lengths_um"]["arm_sig"], ref_um=rep["lengths_um"]["arm_ref"]),
    )
    with open(os.path.join(OUT, "data.json"), "w") as f:
        json.dump(data, f, separators=(",", ":"))
    im = Image.open(os.path.join(HERE, "figures", "parity_layout.png")).convert("RGB")
    w, h = im.size
    im = im.crop((int(w * 0.10), int(h * 0.05), int(w * 0.99), int(h * 0.95)))
    im.thumbnail((1600, 1600))
    im.save(os.path.join(OUT, "layout.jpg"), quality=86)
    print("data.json", os.path.getsize(os.path.join(OUT, "data.json")), "layout.jpg", os.path.getsize(os.path.join(OUT, "layout.jpg")))


if __name__ == "__main__":
    main()
