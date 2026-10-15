#!/usr/bin/env python3
"""
How fast can the paper's interference XOR compute? Both XOR circuits are simulated in the time domain
from their SiEPIC-extracted layouts (td.py), with independent pseudo-random bit streams on the two
inputs and the result read straight off the output, with no electronics in between.

  michelson   XOR 1 of the drawn full adder (klayout/EBeam_heaters_atharvdubey_FullAdder): the
              paper's Michelson, loop mirrors at the ends of the arms
  single_pass klayout/EBeam_heaters_atharvdubey_FastXOR: the same two-beam XOR, arms recombined at a
              second coupler, so each input modulator is crossed once

The input switches are treated as fast phase modulators of the drawn geometry (in the EBeam process
they are heaters; a process with depletion or lithium-niobate modulators is assumed). "wire" runs
send data on A only, with B off: the same light path with no logic, the reference for any penalty the
logic adds. Receiver: photodiode (1 A/W) and a 20 pA/rtHz amplifier, 10 mW laser into the chip.

A rate counts as working when the eye at its centre gives Q >= 7 (bit error rate <= 1e-12) and the eye
is open for at least 30% of the bit period (timing margin for clock recovery).

Outputs: speed_results.json, speed_eyes.npz (eye traces for the figures).
"""
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import jax.numpy as jnp  # noqa: E402
from td import TDCircuit, F, _wg  # noqa: E402
from eyes import experiment, latency, michelson_setup, fast_setup  # noqa: E402
from fast_xor_cal import FastXOR, ROLES  # noqa: E402

RATES_IDEAL = [10e9, 25e9, 50e9, 100e9, 200e9, 400e9]
RATES_IO = [25e9, 50e9, 75e9, 100e9, 125e9, 150e9, 200e9]
IO = {"silicon_50GHz": 50e9, "lithium_niobate_100GHz": 100e9}


def spectral(fx, cal, ch):
    """On/off ratio of each XOR across wavelength (frequency domain, all components dispersive)."""
    wl = np.linspace(1.50, 1.60, 4001)
    p = lambda z: np.abs(z) ** 2  # noqa: E731

    def fast(a, b):
        ph = {"A_mod": cal["A_mod_on" if a else "A_mod_off"], "B_mod": cal["B_mod_on" if b else "B_mod_off"],
              "biasA": cal["biasA"], "biasB": cal["biasB"]}
        return np.asarray(fx.circ(wl=jnp.asarray(wl), **{k + "_ps": {"phi": jnp.full(wl.size, ph[k])} for k in ROLES})["laser1", "xor1"])

    def mich(a, b):
        phs = ch.phases(np.array([a]), np.array([b]), np.array([0]), np.array([0]))
        kw = {k + "_ps": {"phi": jnp.full(wl.size, float(v[0]) / (2 if k in F.BIASES else 1))} for k, v in phs.items()}
        return np.asarray(ch.circ(wl=jnp.asarray(wl), **kw)["laser1", "xor1"])

    out = {}
    for name, f in (("single_pass", fast), ("michelson", mich)):
        s = {ab: f(*ab) for ab in ((0, 0), (0, 1), (1, 0), (1, 1))}
        ratio = 10 * np.log10(np.minimum(p(s[(1, 0)]), p(s[(0, 1)])) / np.maximum(p(s[(1, 1)]), p(s[(0, 0)])))
        out[name] = {f"+-{d}nm": round(float(ratio[np.abs(wl - 1.55) <= d * 1e-3 + 1e-9].min()), 1) for d in (1, 2, 5, 10, 20)}
        out[name]["wavelength_nm"] = (wl * 1e3)[::20].round(2).tolist()
        out[name]["on_off_dB"] = ratio[::20].round(2).tolist()
    return out


def main():
    t0 = time.time()
    fx = FastXOR()
    cal = fx.calibrate()
    ch = F.Chip()
    ch.cal = json.load(open(os.path.join(HERE, "..", "full_adder_results.json")))["majority"]["calibration"]
    circuits = {"michelson": (TDCircuit(), michelson_setup()),
                "single_pass": (TDCircuit(extract=os.path.join(HERE, "..", "klayout", "fast_xor_extracted_netlist.json")), fast_setup(cal))}
    res = {"method": __doc__.strip().splitlines()[0], "fast_xor_calibration": cal, "fast_xor_truth_mW_per_W": fx.truth()}
    res["spectral"] = spectral(fx, cal, ch)
    lat = {k: latency(tc, st) for k, (tc, st) in circuits.items()}
    _, tg = _wg(985.143)            # single-pass: waveguide from the coupler to the XOR grating coupler
    lat["single_pass"]["coupler_to_fibre_ps"] = round(tg * 1e12, 2)
    lat["single_pass"]["modulator_to_coupler_ps"] = round(lat["single_pass"]["first_ps"] - tg * 1e12, 2)
    res["latency"] = lat
    print("latency", lat, f"{time.time() - t0:.0f}s")
    eyes = {}
    runs = []
    for name, (tc, st) in circuits.items():
        for rate in RATES_IDEAL:
            for mode in ("wire", "xor"):
                r, folds, span = experiment(tc, rate, st, mode=mode)
                r.update(circuit=name, io="ideal")
                runs.append(r)
                eyes[f"{name}|ideal|{mode}|{int(rate / 1e9)}"] = (folds, span)
                print(name, "ideal", int(rate / 1e9), mode, round(r["eye_opening_pct"], 1), round(r["Q"], 1), f"{time.time() - t0:.0f}s")
    for io, bw in IO.items():
        for name, (tc, st) in circuits.items():
            if name == "michelson" and io != "silicon_50GHz":
                continue
            for rate in RATES_IO:
                for mode in ("wire", "xor"):
                    r, folds, span = experiment(tc, rate, st, mode=mode, bw_mod=bw, bw_rx=bw)
                    r.update(circuit=name, io=io)
                    runs.append(r)
                    eyes[f"{name}|{io}|{mode}|{int(rate / 1e9)}"] = (folds, span)
                    print(name, io, int(rate / 1e9), mode, round(r["eye_opening_pct"], 1), round(r["Q"], 1), f"{time.time() - t0:.0f}s")
    res["runs"] = runs
    # summary: fastest rate with Q >= 7 (BER <= 1e-12) and the penalty of the logic over the wire
    summ = {}
    for name in circuits:
        for io in ["ideal"] + list(IO):
            rs = [r for r in runs if r["circuit"] == name and r["io"] == io]
            if not rs:
                continue
            ok = {m: [r["rate_Gbps"] for r in rs if r["mode"] == m and r["Q"] >= 7 and r["correct"] and r["eye_width_pct"] >= 30]
                  for m in ("wire", "xor")}
            pen = []
            for r in rs:
                if r["mode"] != "xor":
                    continue
                w = next(q for q in rs if q["mode"] == "wire" and q["rate_Gbps"] == r["rate_Gbps"])
                if r["eye_opening_pct"] > 0 and w["eye_opening_pct"] > 0:
                    pen.append(dict(rate_Gbps=r["rate_Gbps"], xor_vs_wire_dB=round(10 * np.log10(w["eye_opening_pct"] / r["eye_opening_pct"]), 2)))
            summ[f"{name}|{io}"] = dict(max_rate_wire_Gbps=max(ok["wire"], default=0), max_rate_xor_Gbps=max(ok["xor"], default=0),
                                        logic_penalty=pen)
    res["summary"] = summ
    for k, v in summ.items():
        print(k, v["max_rate_wire_Gbps"], v["max_rate_xor_Gbps"], [p["xor_vs_wire_dB"] for p in v["logic_penalty"]])
    with open(os.path.join(HERE, "speed_results.json"), "w") as f:
        json.dump(res, f, indent=1, default=float)
    def small(f):                     # the figures plot 160 traces; 400 points per two-bit trace is plenty
        f = f[:160]
        return f[:, np.linspace(0, f.shape[1] - 1, 400).round().astype(int)].astype(np.float32)

    np.savez_compressed(os.path.join(HERE, "speed_eyes.npz"), **{k.replace("|", "__"): small(v[0]) for k, v in eyes.items()},
                        **{"span__" + k.replace("|", "__"): np.array(v[1]) for k, v in eyes.items()})
    print(f"done {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
