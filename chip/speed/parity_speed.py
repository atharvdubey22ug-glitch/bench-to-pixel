#!/usr/bin/env python3
"""
How fast can the phase-encoded parity chip compute? Eight independent random bit streams drive the eight
phase sections of klayout/EBeam_heaters_atharvdubey_Parity8 and the parity of all eight is read off
the PARITY grating coupler, in the time domain (td.py: the SiEPIC extraction as a delay-line scattering
network, the same model as speed_study.py).

The sections sit 150 um apart along the arm, so light reaches section 8 about 14.7 ps after section 1.
  aligned    each input is delayed by the light's travel time to its section (a fixed length of
             electrical line per input), so the k-th bit of every stream acts on the same slice of light
  unaligned  all eight inputs launched at the same instant: the skew is left in
  wire       data on section 1 only, the other seven held at 0: the same light path, no logic, the
             reference for any penalty the logic adds
  aligned_err5  aligned, with each section's '1' phase off by a random 5% (driver swing not trimmed)

Sections are treated as fast phase modulators of the drawn placement (in the EBeam process they are
heaters; a process with modulators is assumed, as in speed_study.py). Receiver as in speed_study.py:
photodiode 1 A/W, 20 pA/rtHz amplifier, 10 mW laser. A rate counts as working at Q >= 7 and eye width
>= 30%.

Outputs: parity_speed_results.json, parity_eyes.npz.
"""
import json
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, ".."))
from td import TDCircuit, _wg  # noqa: E402
from eyes import analyse, drive, latency  # noqa: E402
import parity_sim as PS  # noqa: E402

EXTRACT = os.path.join(HERE, "..", "klayout", "parity_extracted_netlist.json")
RATES_IDEAL = [25e9, 50e9, 100e9, 200e9, 400e9]
RATES_IO = [25e9, 50e9, 75e9, 100e9, 125e9, 150e9, 200e9]
IO = {"ideal": None, "silicon_50GHz": 50e9, "lithium_niobate_100GHz": 100e9}
MODES = ["wire", "aligned", "unaligned"]


def bias_phase():
    c = PS.Circuit()
    th = np.linspace(0, 2 * np.pi, 7201)
    return float(th[np.argmin(c.power({"bias": th})["parity"])])


def section_delays(tc):
    """Steps from each section to the end of the arm, for light travelling forward (laser -> coupler);
    the network also holds the backward link through the same sections, which is skipped."""
    by_link = {}
    for i, r, d in tc.heat_terms:
        if r in PS.SECT:
            by_link.setdefault(i, {})[r] = d
    fwd = [v for v in by_link.values() if v["S1"] > v["S8"]]
    assert len(fwd) == 1, by_link
    return fwd[0]


def run(tc, rate, mode, bias, bw=None, nbits=254, seed=11, err=None):
    dt = tc.dt
    spb = 1.0 / rate / dt
    lead = int(round(300e-12 / dt))
    dly = section_delays(tc)
    n = lead + int(math.ceil((nbits + 6) * spb)) + max(dly.values())
    rng = np.random.default_rng(seed)
    bits = rng.integers(0, 2, (PS.N, nbits))
    if mode == "wire":
        bits[1:] = 0
    ph = {"bias": bias}
    gain = np.ones(PS.N) if err is None else 1 + err * np.random.default_rng(seed + 1).standard_normal(PS.N)
    for k, r in enumerate(PS.SECT):
        skew = dly["S1"] - dly[r] if mode != "unaligned" else 0
        ph[r] = np.pi * gain[k] * drive(bits[k], rate, dt, n, lead + skew, bw)
    e = tc.run(n, ph, tc.fibre["laser1"], np.full(n, 1 + 0j), ["parity"])["parity"]
    want = bits.sum(0) % 2
    r, folds, span = analyse(e, want, lead, spb, dt, rate, bw)
    r.update(mode=mode if err is None else f"{mode}_err{int(err * 100)}", bw_mod_GHz=bw / 1e9 if bw else None)
    return r, folds, span


def main():
    t0 = time.time()
    tc = TDCircuit(extract=EXTRACT)
    bias = bias_phase()
    dly = section_delays(tc)
    res = {"method": __doc__.strip().splitlines()[0], "bias_rad": bias,
           "section_skew_ps": {r: round((dly["S1"] - d) * tc.dt * 1e12, 2) for r, d in sorted(dly.items())}}
    base = {"bias": bias}
    lat = {}
    for r in ("S1", "S8"):
        lat[r] = latency(tc, (base, {"A": (r, np.pi, 0.0)}), out_port="parity")
    _, tg = _wg(json.load(open(os.path.join(HERE, "..", "klayout", "parity_build_report.json")))["lengths_um"]["par_out"])
    lat["coupler_to_fibre_ps"] = round(tg * 1e12, 2)
    res["latency"] = lat
    print("skew", res["section_skew_ps"], "latency", lat, f"{time.time() - t0:.0f}s", flush=True)
    runs, eyes = [], {}
    for io, bw in IO.items():
        for rate in (RATES_IDEAL if bw is None else RATES_IO):
            for mode in MODES:
                r, folds, span = run(tc, rate, mode, bias, bw)
                r.update(io=io)
                runs.append(r)
                eyes[f"{io}|{mode}|{int(rate / 1e9)}"] = (folds, span)
                print(io, int(rate / 1e9), mode, round(r["eye_opening_pct"], 1), round(r["Q"], 1), r["eye_width_pct"], f"{time.time() - t0:.0f}s", flush=True)
    for rate in (50e9, 100e9):
        r, folds, span = run(tc, rate, "aligned", bias, 100e9, err=0.05)
        r.update(io="lithium_niobate_100GHz")
        runs.append(r)
        eyes[f"lithium_niobate_100GHz|aligned_err5|{int(rate / 1e9)}"] = (folds, span)
        print("err5", int(rate / 1e9), round(r["eye_opening_pct"], 1), round(r["Q"], 1), flush=True)
    res["runs"] = runs
    summ = {}
    for io in IO:
        rs = [r for r in runs if r["io"] == io]
        summ[io] = {m: max([r["rate_Gbps"] for r in rs if r["mode"] == m and r["Q"] >= 7 and r["correct"] and r["eye_width_pct"] >= 30], default=0)
                    for m in MODES}
        pen = []
        for r in rs:
            if r["mode"] != "aligned":
                continue
            w = next(q for q in rs if q["mode"] == "wire" and q["rate_Gbps"] == r["rate_Gbps"])
            if r["eye_opening_pct"] > 0 and w["eye_opening_pct"] > 0:
                pen.append(dict(rate_Gbps=r["rate_Gbps"], parity_vs_wire_dB=round(10 * np.log10(w["eye_opening_pct"] / r["eye_opening_pct"]), 2)))
        summ[io]["logic_penalty"] = pen
    res["summary"] = summ
    print(json.dumps(summ, indent=0))
    with open(os.path.join(HERE, "parity_speed_results.json"), "w") as f:
        json.dump(res, f, indent=1, default=float)

    def small(f):
        f = f[:160]
        return f[:, np.linspace(0, f.shape[1] - 1, 400).round().astype(int)].astype(np.float32)

    np.savez_compressed(os.path.join(HERE, "parity_eyes.npz"), **{k.replace("|", "__"): small(v[0]) for k, v in eyes.items()},
                        **{"span__" + k.replace("|", "__"): np.array(v[1]) for k, v in eyes.items()})
    print(f"done {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
