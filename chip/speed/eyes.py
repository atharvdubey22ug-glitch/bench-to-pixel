#!/usr/bin/env python3
"""Drive the XOR's two input switches with independent bit streams and measure the output eye."""
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from td import TDCircuit, F  # noqa: E402

CAL = json.load(open(os.path.join(HERE, "..", "full_adder_results.json")))["majority"]["calibration"]


def latency(tc, setup, key="A", out_port="xor", src_port="laser1", n=20000):
    """Time from a change at an input modulator to its first effect at the output fibre, and the time
    between the first and last significant response (a second pass, if any)."""
    base, sw = setup
    role, on, off = sw[key]
    ph = dict(base)
    ph[role] = on
    src = np.ones(n, complex)
    ref = np.abs(tc.run(n, ph, tc.fibre[src_port], src, [out_port])[out_port]) ** 2
    k0 = n // 2
    pul = np.full(n, on)
    pul[k0:k0 + 10] += 0.3
    ph[role] = pul
    o = np.abs(tc.run(n, ph, tc.fibre[src_port], src, [out_port])[out_port]) ** 2
    d = np.abs(o - ref)
    idx = np.where(d > 0.05 * d.max())[0]
    groups = np.split(idx, np.where(np.diff(idx) > 5)[0] + 1)
    t = [round((g[0] - k0) * tc.dt * 1e12, 2) for g in groups]
    strong = [round((g[0] - k0) * tc.dt * 1e12, 2) for g in groups if d[g].max() > 0.3 * d.max()]
    return dict(first_ps=t[0], responses_ps=t, strong_responses_ps=strong)


def prbs(order, n, seed):
    """PRBS-7 style maximal-length sequence (x^7 + x^6 + 1), n bits, starting state from seed."""
    taps = {7: (7, 6), 9: (9, 5)}[order]
    s = (seed % (2 ** order - 1)) + 1
    out = []
    for _ in range(n):
        b = ((s >> (taps[0] - 1)) ^ (s >> (taps[1] - 1))) & 1
        s = ((s << 1) | b) & (2 ** order - 1)
        out.append(b)
    return np.array(out)


def gfilter(x, dt, bw):
    """Gaussian low-pass with 3 dB bandwidth bw (Hz); bw=None: no filter."""
    if bw is None:
        return x
    f = np.fft.fftfreq(x.size, dt)
    return np.fft.ifft(np.fft.fft(x) * np.exp(-np.log(2) * (f / bw) ** 2 / 2)).real if np.isrealobj(x) else \
        np.fft.ifft(np.fft.fft(x) * np.exp(-np.log(2) * (f / bw) ** 2 / 2))


def drive(bits, rate, dt, n_steps, start, bw_mod):
    """NRZ drive level 0..1 sampled every dt, bits starting at step `start`, modulator bandwidth bw_mod."""
    t = (np.arange(n_steps) - start) * dt
    k = np.floor(t * rate).astype(int)
    lvl = np.where((k >= 0) & (k < bits.size), bits[np.clip(k, 0, bits.size - 1)], 0).astype(float)
    return np.clip(gfilter(lvl, dt, bw_mod), 0, 1)


def michelson_setup(cal=CAL):
    """Per-pass heater phases of the drawn full adder's XOR 1 with every input off, and the on/off
    phases of its two input switches."""
    ch = F.Chip.__new__(F.Chip)
    ch.cal = cal
    ph = ch.phases(np.array([0]), np.array([0]), np.array([0]), np.array([0]))
    base = {k: float(v[0]) / (2.0 if k in F.BIASES else 1.0) for k, v in ph.items()}
    return base, {"A": ("A_x1", cal["A_x1_on"], cal["A_x1_off"]), "B": ("B_x1", cal["B_x1_on"], cal["B_x1_off"])}


def fast_setup(cal):
    base = {"A_mod": cal["A_mod_off"], "B_mod": cal["B_mod_off"], "biasA": cal["biasA"], "biasB": cal["biasB"]}
    return base, {"A": ("A_mod", cal["A_mod_on"], cal["A_mod_off"]), "B": ("B_mod", cal["B_mod_on"], cal["B_mod_off"])}


def experiment(tc, rate, setup, nbits=254, bw_mod=None, bw_rx=None, p_laser=10e-3, tia_pA=20.0, mode="xor",
               out_port="xor", src_port="laser1"):
    """mode 'xor': independent bit streams on A and B. mode 'wire': data on A, B held off (the same
    path with no logic, the reference for the penalty the gate adds)."""
    base, sw = setup
    dt = tc.dt
    spb = 1.0 / rate / dt
    lead = int(round(300e-12 / dt))                       # let the circuit settle before the data starts
    n = lead + int(math.ceil((nbits + 6) * spb))
    a_bits = prbs(7, nbits, 1)
    b_bits = prbs(7, nbits, 50) if mode == "xor" else np.zeros(nbits, int)
    per = dict(base)

    def wave(key, bits):
        role, on, off = sw[key]
        delta = ((on - off + np.pi) % (2 * np.pi)) - np.pi
        return role, off + delta * drive(bits, rate, dt, n, lead, bw_mod)

    for key, bits in (("A", a_bits), ("B", b_bits)):
        role, w = wave(key, bits)
        per[role] = w
    src = np.full(n, 1 + 0j)
    e = tc.run(n, per, tc.fibre[src_port], src, [out_port])[out_port]
    i = F.DET["resp"] * p_laser * np.abs(e) ** 2              # photocurrent, A
    i = gfilter(i, dt, bw_rx)
    want = a_bits ^ b_bits
    # latency: align the output with the expected bits (search a few bit periods of delay)
    dlys = np.arange(0, int(4 * spb) + int(200e-12 / dt), max(1, int(spb / 40)))
    cs = []
    for dly in dlys:
        idx = (lead + dly + (np.arange(10, nbits - 4) + 0.5) * spb).astype(int)
        idx = idx[idx < n]
        w = want[10:10 + idx.size]
        cs.append(np.corrcoef(i[idx], w)[0, 1] if w.std() > 0 else 0)
    cs = np.array(cs)
    k0 = int(np.argmax(cs))                     # with sharp edges many delays tie: take the middle of the tie
    k1 = k0
    while k1 + 1 < cs.size and cs[k1 + 1] >= cs[k0] - 1e-6:
        k1 += 1
    dly = int(dlys[(k0 + k1) // 2])
    # eye: scan the sampling time over three bit periods around the alignment; bit k is compared with
    # the bit that should be there. The open stretch (every 1 above every 0) is the eye; its length is
    # the eye width and the receiver samples in the middle of its best part.
    bits_idx = np.arange(12, nbits - 8)
    phases = np.arange(-1.0, 2.0, 1.0 / 40)
    res = []
    for phs in phases:
        idx = (lead + dly + (bits_idx + phs) * spb).astype(int)
        ok = (idx < n) & (idx >= 0)
        v, w = i[idx[ok]], want[bits_idx[ok]]
        lo1, hi0 = v[w == 1].min(), v[w == 0].max()
        res.append((lo1 - hi0, phs, lo1, hi0, v[w == 1].mean(), v[w == 0].mean()))
    op = np.array([r[0] for r in res])
    j = int(np.argmax(op))
    if op[j] > 0:
        a_, b_ = j, j
        while a_ > 0 and op[a_ - 1] > 0:
            a_ -= 1
        while b_ + 1 < op.size and op[b_ + 1] > 0:
            b_ += 1
        width = min(1.0, (b_ - a_ + 1) / 40)
        top = [k for k in range(a_, b_ + 1) if op[k] >= op[j] - 0.1 * op[j]]
        mid = top[len(top) // 2]
    else:
        width, mid = 0.0, j
    open_, phs, lo1, hi0, m1, m0 = res[mid]
    bw_noise = bw_rx if bw_rx else 0.75 * rate
    sig = lambda x: math.sqrt((2 * F.P.QE * max(x, 0) + (tia_pA * 1e-12) ** 2) * bw_noise)  # noqa: E731
    q = (lo1 - hi0) / (sig(lo1) + sig(hi0))
    ber = 0.5 * math.erfc(q / math.sqrt(2)) if q > 0 else 0.5
    # trace for plotting: two bit periods per fold
    t0 = lead + dly + (phs - 0.5) * spb                  # centre the plotted eye on the sampling point
    seg = int(round(2 * spb))
    folds = [i[int(t0 + k * spb) - seg // 4: int(t0 + k * spb) - seg // 4 + seg] for k in range(10, nbits - 6)]
    folds = np.array([f for f in folds if f.size == seg])
    return dict(rate_Gbps=rate / 1e9, mode=mode, bw_mod_GHz=bw_mod / 1e9 if bw_mod else None, bw_rx_GHz=bw_rx / 1e9 if bw_rx else None,
                eye_opening_pct=100 * open_ / m1, eye_width_pct=100 * width, one_uA=m1 * 1e6, zero_uA=m0 * 1e6, Q=q, BER=ber,
                sample_phase=phs, correct=bool(open_ > 0)), folds * 1e6, seg * dt * 1e12
