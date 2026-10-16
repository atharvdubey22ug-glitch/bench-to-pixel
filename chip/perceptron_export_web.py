#!/usr/bin/env python3
"""Export the optical neuron for the web page (../perceptron/data.json, ../perceptron/layout.jpg).

For each process corner: the linear field model of the drawn circuit (t, the field each of the 17
waveguides delivers to each output), and the first simulated die of perceptron_sim.py (heater
efficiencies, arm offsets, crosstalk, calibration), so the page computes the chip's outputs for any
input with the same arithmetic as perceptron_sim.Die + Circuit.lin_fields. Plus the speed study's
eye traces and summary.
"""
import base64
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import full_adder_sim as F  # noqa: E402
import perceptron_sim as PS  # noqa: E402

OUT = os.path.join(HERE, "..", "perceptron")
TRACES, POINTS = 48, 100


def cx(a):
    return [[round(float(v.real), 7), round(float(v.imag), 7)] for v in a]


def main(seed=11):
    os.makedirs(OUT, exist_ok=True)
    res = json.load(open(os.path.join(HERE, "perceptron_results.json")))
    sp = json.load(open(os.path.join(HERE, "speed", "perceptron_speed_results.json")))
    rep = json.load(open(os.path.join(HERE, "klayout", "perceptron_build_report.json")))
    X = PS.crosstalk_matrix(PS.geometry())
    corners, check = {}, {}
    for corner in F.CORNERS:
        c = PS.Circuit(corner)
        die = PS.Die(c, np.random.default_rng(seed * 1000 + int(corner[0]) * 10 + int(corner[1]) + 0), X)   # die 0 of the study
        cal = PS.calibrate(die)
        key = f"{int(corner[0])}nm_x_{int(corner[1])}nm"
        r = res["corners"][key]
        corners[key] = dict(t_plus=cx(c.t["plus"]), t_minus=cx(c.t["minus"]), eta=(die.eta * 1e-3).round(7).tolist(),   # rad per mW
                            phi0=die.phi0.round(7).tolist(), p_plus_mW=(cal["p_plus"] * 1e3).round(5).tolist(),
                            p_minus_mW=(cal["p_minus"] * 1e3).round(5).tolist(),
                            worst_max_error_levels=r["worst_max_error_levels"], worst_agree_with_ideal_pct=r["worst_agree_with_ideal_pct"],
                            linear_model_error_levels=r["worst_linear_model_error_levels"], dies=r["dies"])
        if corner == F.NOMINAL:
            st = np.array([[1] * 16, [-1] * 16, [1, -1] * 8])
            check = dict(states=st.tolist(), D=PS.diff_of(die.fields(PS.powers_for(cal, st), full=False)).round(9).tolist())
    z = np.load(os.path.join(HERE, "speed", "perceptron_eyes.npz"))
    full = max(float(np.abs(z[k]).max()) for k in z.files) * 1.02
    eyes = {}
    for run in sp["runs"]:
        key = f"{run['io']}|{run['mode']}|{int(run['rate_Gbps'])}"
        f = z[key.replace("|", "__")][:TRACES]
        f = f[:, np.linspace(0, f.shape[1] - 1, POINTS).round().astype(int)]
        q = np.clip(np.round((f / full + 1) / 2 * 255), 0, 255).astype(np.uint8)       # -full..+full -> 0..255
        eyes[key] = dict(b64=base64.b64encode(q.tobytes()).decode(), n=int(q.shape[0]),
                         opening=round(run["level_opening"], 3), width=round(run["level_width"], 3),
                         Q=None if run["threshold_Q"] != run["threshold_Q"] else round(run["threshold_Q"], 1),
                         decision_ok=run["threshold_correct"], levels=run["levels_present"], level_uA=round(run["level_uA"], 1))
    data = dict(
        roles=PS.ROLES, font=PS.FONT, arm_of_pixel=PS.ARM_OF_PIXEL, bias_arm=PS.BIAS_ARM, X=X.round(6).tolist(),
        resp=F.DET["resp"], corners=corners, check=check,
        results=dict(die_model=res["die_model"], example=dict(max_error_levels=res["example_nominal"]["max_error_levels"],
                                                              without_crosstalk=res["example_nominal"]["max_error_levels_without_crosstalk"]),
                     laser_uW=res["laser_uW_for_one_level_at_Q7"], spectral={k: v for k, v in res["spectral"].items() if k.startswith("+-")},
                     digits_nominal=res["corners"]["220nm_x_500nm"]["digits_example"], distances=res["digit_distances"]),
        speed=dict(summary=sp["summary"], skew_ps=sp["skew_ps"], latency_ps=sp["latency"]["first_ps"],
                   coupler_to_fibre_ps=sp["latency"]["coupler_to_fibre_ps"], digit=sp["digit"], threshold=sp["threshold"],
                   eyes=eyes, points=POINTS, full_uA=full),
        chip=dict(die_um=rep["die_um"], components=rep["components"], errors=rep["siepic_verification_errors"],
                  arm_um=rep["arm_lengths_um"][0], signal_um=rep["signal_path_um"], ref_um=rep["reference_path_um"]))
    with open(os.path.join(OUT, "data.json"), "w") as f:
        json.dump(data, f, separators=(",", ":"))
    from PIL import Image
    im = Image.open(os.path.join(HERE, "figures", "perceptron_layout.png")).convert("RGB")
    im.thumbnail((1500, 1500))
    im.save(os.path.join(OUT, "layout.jpg"), quality=86)
    print("data.json", os.path.getsize(os.path.join(OUT, "data.json")), "check", check["D"])


if __name__ == "__main__":
    main()
