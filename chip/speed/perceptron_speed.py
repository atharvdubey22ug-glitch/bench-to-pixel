#!/usr/bin/env python3
"""
How fast can the optical neuron classify? A stream of 3x5-pixel digit images, one image per bit slot
with all 15 pixels in parallel (plus the bias arm), drives the 16 phase sections of
klayout/EBeam_heaters_atharvdubey_Perceptron16; the weights are fixed to the stored "7". The balanced
output current is read straight off the chip (time domain, td.py: the SiEPIC extraction as a
delay-line scattering network, as in speed_study.py and parity_speed.py).

Images: random digits with 0, 1 or 2 pixels flipped, so the sum S = 16 - 2 * (pixels differing
from the stored 7) takes many values. Two measures, both against the ideal sum of each slot:
  levels   every value of S that occurs must stay apart from the next one at the sampling instant:
           opening = smallest gap between neighbouring levels / one level, at the best instant
  decision the neuron fires when S >= 12 (at most two pixels differ from a 7): Q of that threshold,
           with shot noise and a 20 pA/rtHz amplifier on the balanced pair, 10 mW laser
Drives: the heaters form a staircase, so light passes the arms' sections at different points of
their (equal) paths; "aligned" delays each input by its section's position, "unaligned" does not.
Sections are treated as fast phase modulators of the drawn placement, with ideal, 100 GHz or 50 GHz
bandwidth, and the same bandwidth on the photodiodes. "wire": one input toggling, the rest held.

Outputs: perceptron_speed_results.json, perceptron_eyes.npz.
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
from td import TDCircuit, F, _wg  # noqa: E402
from eyes import drive, gfilter, latency  # noqa: E402
import perceptron_sim as PS  # noqa: E402

EXTRACT = PS.EXTRACT
RATES_IDEAL = [25e9, 50e9, 100e9, 200e9, 400e9]
RATES_IO = [25e9, 50e9, 75e9, 100e9, 125e9, 150e9, 200e9]
IO = {"ideal": None, "silicon_50GHz": 50e9, "lithium_niobate_100GHz": 100e9}
MODES = ["aligned", "unaligned", "wire"]
DIGIT, THRESH = 7, 12
P_LASER, TIA = 10e-3, 20e-12


def align_phases():
    """Phase of each arm that puts its light in phase with the reference at the output (nominal
    chip, ideal calibration through the linear model of perceptron_sim)."""
    c = PS.Circuit()
    ph = np.zeros(len(PS.ROLES))
    th = np.linspace(0, 2 * np.pi, 1441)
    for _ in range(2):
        for j in range(PS.N):
            P = np.tile(ph, (th.size, 1))
            P[:, j] = th
            ph[j] = th[np.argmax(PS.diff_of(c.lin_fields(P)))]
    return ph


def section_delays(tc):
    """Steps from each arm's section to the combiner end of its arm (forward link only)."""
    comps = {c["idx"]: c for c in json.load(open(EXTRACT))["components"]}
    xmid = np.mean([c["center"][0] for c in comps.values() if c["kind"] == "wg_heater" and c["role"] != "REF"])
    out = {}
    for i, r, d in tc.heat_terms:
        if r == "REF":
            continue
        q_comp = comps[tc.ports[tc.links[i]["q"]][0]]
        if q_comp["center"][0] > xmid:                 # link ends in the combiner tree: forward
            out[r] = d
    assert len(out) == PS.N, out
    return out


def images(n, rng):
    imgs, lab = [], []
    for _ in range(n):
        d = int(rng.integers(0, 10))
        v = np.array([1 if ch == "1" else -1 for ch in PS.FONT[d]])
        for i in rng.choice(15, int(rng.integers(0, 3)), replace=False):
            v[i] *= -1
        imgs.append(v)
        lab.append(d)
    return np.array(imgs), np.array(lab)


def run(tc, rate, mode, align, dly, bw=None, nimg=254, seed=5):
    dt = tc.dt
    spb = 1.0 / rate / dt
    lead = int(round(300e-12 / dt))
    dmax = max(dly.values())
    n = lead + int(math.ceil((nimg + 6) * spb)) + dmax + int(round(400e-12 / dt))   # tail: the chip's own latency
    rng = np.random.default_rng(seed)
    W = PS.weights()[DIGIT]
    if mode == "wire":
        x = np.tile(W, (nimg, 1))
        x[:, 1] = rng.choice([-1, 1], nimg)          # one pixel arm toggling, the rest agree with the 7
    else:
        imgs, _ = images(nimg, rng)
        x = PS.arm_inputs(imgs)
    states = x * W[None, :]                          # +1: arm in phase with the reference
    S = states.sum(1)
    ph = {"REF": align[-1]}
    for k, r in enumerate(PS.ARMS):
        skew = dmax - dly[r] if mode != "unaligned" else 0
        b = (states[:, k] < 0).astype(int)
        ph[r] = align[k] + np.pi * drive(b, rate, dt, n, lead + skew, bw)
    e = tc.run(n, ph, tc.fibre["laser1"], np.full(n, 1 + 0j), ["plus", "minus"])
    ip = F.DET["resp"] * P_LASER * gfilter(np.abs(e["plus"]) ** 2, dt, bw)
    im = F.DET["resp"] * P_LASER * gfilter(np.abs(e["minus"]) ** 2, dt, bw)
    i = ip - im
    return analyse(i, ip + im, S, lead, spb, dt, rate, bw)


def analyse(i, isum, S, lead, spb, dt, rate, bw):
    nimg, n = S.size, i.size
    # latency: align the output with S (correlation over delays)
    dlys = np.arange(0, int(4 * spb) + int(300e-12 / dt), max(1, int(spb / 40)))
    cs = []
    for dly in dlys:
        idx = (lead + dly + (np.arange(10, nimg - 4) + 0.5) * spb).astype(int)
        idx = idx[idx < n]
        cs.append(np.corrcoef(i[idx], S[10:10 + idx.size])[0, 1])
    k0 = int(np.argmax(cs))
    k1 = k0
    while k1 + 1 < len(cs) and cs[k1 + 1] >= cs[k0] - 1e-6:
        k1 += 1
    dly = int(dlys[(k0 + k1) // 2])
    slots = np.arange(12, nimg - 8)
    levels = sorted(set(S[slots]))
    # gain: one level in current, from a straight-line fit at the centre of the bit
    idx_c = (lead + dly + (slots + 0.5) * spb).astype(int)
    g, off = np.polyfit(S[slots], i[idx_c], 1)
    lvl = 2 * g
    res = []
    for phs in np.arange(-1.0, 2.0, 1.0 / 40):
        idx = (lead + dly + (slots + phs) * spb).astype(int)
        ok = (idx >= 0) & (idx < n)
        v, s = i[idx[ok]], S[slots[ok]]
        gaps = [(v[s == b].min() - v[s == a].max()) for a, b in zip(levels[:-1], levels[1:]) if b - a == 2]
        lo = v[s >= THRESH].min() if (s >= THRESH).any() else np.nan
        hi = v[s < THRESH].max() if (s < THRESH).any() else np.nan
        res.append((min(gaps) / lvl if gaps else 0, phs, lo, hi, idx[ok], s))
    op = np.array([r[0] for r in res])
    j = int(np.argmax(op))
    width = 0.0
    if op[j] > 0:
        a_, b_ = j, j
        while a_ > 0 and op[a_ - 1] > 0:
            a_ -= 1
        while b_ + 1 < op.size and op[b_ + 1] > 0:
            b_ += 1
        width = min(1.0, (b_ - a_ + 1) / 40)
    opening, phs, lo, hi, idx, s = res[j]
    bw_noise = bw if bw else 0.75 * rate
    sig = lambda cur: math.sqrt((2 * F.P.QE * max(cur, 0) + 2 * TIA ** 2) * bw_noise)  # noqa: E731
    isum_s = isum[idx]
    both = not (np.isnan(lo) or np.isnan(hi))
    q = (lo - hi) / (sig(isum_s[s >= THRESH].mean()) + sig(isum_s[s < THRESH].mean())) if both else float("nan")
    t0 = lead + dly + (phs - 0.5) * spb
    seg = int(round(2 * spb))
    folds = [i[int(t0 + k * spb) - seg // 4: int(t0 + k * spb) - seg // 4 + seg] for k in range(10, nimg - 6)]
    folds = np.array([f for f in folds if f.size == seg])
    return dict(rate_Gbps=rate / 1e9, levels_present=[int(v) for v in levels], level_opening=float(opening), level_width=float(width),
                level_uA=float(lvl * 1e6), threshold_Q=float(q), threshold_correct=bool(lo > hi) if both else None,
                sample_phase=float(phs)), folds * 1e6


def main():
    t0 = time.time()
    tc = TDCircuit(extract=EXTRACT)
    align = align_phases()
    dly = section_delays(tc)
    res = {"method": __doc__.strip().splitlines()[0], "digit": DIGIT, "threshold": THRESH,
           "skew_ps": {r: round((max(dly.values()) - d) * tc.dt * 1e12, 2) for r, d in sorted(dly.items())}}
    base = {r: float(align[j]) for j, r in enumerate(PS.ROLES)}
    first = max(dly, key=dly.get)
    lat = latency(tc, (base, {"A": (first, base[first] + np.pi, base[first])}), out_port="plus")
    rep = json.load(open(os.path.join(HERE, "..", "klayout", "perceptron_build_report.json")))
    _, tg = _wg(rep["lengths_um"]["out_plus"])
    lat["coupler_to_fibre_ps"] = round(tg * 1e12, 2)
    lat["first_section"] = first
    res["latency"] = lat
    print("skew", res["skew_ps"], "latency", lat, flush=True)
    runs, eyes = [], {}
    for io, bw in IO.items():
        for rate in (RATES_IDEAL if bw is None else RATES_IO):
            for mode in MODES:
                r, folds = run(tc, rate, mode, align, dly, bw)
                r.update(io=io, mode=mode, bw_GHz=bw / 1e9 if bw else None)
                runs.append(r)
                eyes[f"{io}|{mode}|{int(rate / 1e9)}"] = folds
                print(io, int(rate / 1e9), mode, round(r["level_opening"], 2), round(r["level_width"], 2), round(r["threshold_Q"], 1),
                      f"{time.time() - t0:.0f}s", flush=True)
    res["runs"] = runs
    summ = {}
    for io in IO:
        rs = [r for r in runs if r["io"] == io]
        summ[io] = {m: dict(all_levels=max([r["rate_Gbps"] for r in rs if r["mode"] == m and r["level_opening"] > 0 and r["level_width"] >= 0.3], default=0),
                            decision=max([r["rate_Gbps"] for r in rs if r["mode"] == m and r["threshold_correct"] and r["threshold_Q"] >= 7], default=0))
                    for m in MODES}
    res["summary"] = summ
    print(json.dumps(summ))
    with open(os.path.join(HERE, "perceptron_speed_results.json"), "w") as f:
        json.dump(res, f, indent=1, default=float)

    def small(f):
        f = f[:160]
        return f[:, np.linspace(0, f.shape[1] - 1, 400).round().astype(int)].astype(np.float32)

    np.savez_compressed(os.path.join(HERE, "perceptron_eyes.npz"), **{k.replace("|", "__"): small(v) for k, v in eyes.items()})
    print(f"done {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
