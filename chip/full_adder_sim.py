#!/usr/bin/env python3
"""
Optical full adder chip: circuit simulation of the paper's Figure 15 on the SiEPIC EBeam
(Applied Nanotools) passive silicon process with metal heaters.

Mapping from the bench (Fig. 15/16 of "Using Polarisation and Interference for Optical Computing")
  S1 (1)          Y-branch: light for the carry-in half / the A,B half
  S2 (2), S3 (3)  Y-branches: tap light for input A and input B of AND gate 1
  S4 (8)          broadband directional coupler: Michelson XOR 1 (inputs A, B)
  S5 (10)         Y-branches: light for AND gate 2 and XOR 2
  S6 (13)         broadband directional coupler: Michelson XOR 2 (inputs XOR1, C_in) -> SUM
  mirrors 7,9,11,12   Sagnac loop mirrors (Y-branch + waveguide loop)
  blocking a beam     thermo-optic Mach-Zehnder switch (two directional couplers + heater)
  polarisers (marking) the same switches set to partial transmission
  OR photoresistor    the four AND-branch outputs on photodiodes wired in parallel (currents add)
  Arduino threshold   comparators

One change from Fig. 15 (the "hybrid" fix). XOR 1's light is not fed into XOR 2. A light-only
cascade cannot work: XOR 1 cancels A and B by giving them opposite phases at its dark port, so
"only A" and "only B" leave with opposite phase, and no carry-in beam can cancel both
(cascade_failure() shows this). XOR 1 is therefore detected, thresholded and re-applied to XOR 2
as an input (the X switches), the same way the bench's Arduino applied thresholds.

The whole chip is one SAX scattering network of PDK S-matrices (grating couplers, Y-branches,
broadband couplers, strip waveguides) plus heater phase shifters, so back-reflections are
included. Heaters are calibrated per chip with measurable sweeps (calibrate()).
"""
import json
import math
import os
import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "sim"))
import jax.numpy as jnp  # noqa: E402
import sax  # noqa: E402
from simphony.libraries import siepic  # noqa: E402

import pdk_sim as P  # noqa: E402  (PDK access, receiver helpers)

LAM = 1550e-9
NOMINAL = (220.0, 500.0)
CORNERS = [(t, w) for t in (210.0, 220.0, 230.0) for w in (480.0, 500.0, 520.0)]

# Off-chip detection (passive chip): InGaAs photodiodes behind the fibre array, transimpedance
# amplifiers and comparators. Heater switches settle in tens of microseconds, so 1 MHz is ample.
DET = dict(resp=1.0, dark=1e-9, tia_pA=2.0, bw=1e6, rin_dB=-140.0)
LOSS = P.WG_LOSS_DB_CM

# Waveguide lengths (um). Taken from the drawn layout (layout_extract.json, written by layout.py)
# when it exists; otherwise these schematic values are used.
LEN = dict(gc_s1=60, in1=500, in2=500, ab1=75, ab2=75, sx1=215, sx2=215, x_out1=950, x_out2=950,
           arm_a1=636, arm_b1=636, arm_a2=636, arm_b2=636, mzi_arm=250, loop=94,
           and_in_A_and=155, and_in_B_and=375, and_in_C_and=155, and_in_X_and=375,
           and_out_A_and=80, and_out_B_and=80, and_out_C_and=80, and_out_X_and=80)
_EXTRACT = os.path.join(HERE, "layout_extract.json")
if os.path.exists(_EXTRACT):
    with open(_EXTRACT) as _f:
        LEN = {**LEN, **json.load(_f)["lengths_um"]}

SWITCHES = ["A_and", "B_and", "C_and", "X_and", "A_x1", "B_x1", "C_x2", "X_x2"]
BIASES = ["bias1", "bias2"]
OUTPUTS = ["xor1", "sum", "c_a", "c_b", "c_c", "c_d"]


def _phase(wl=1.55, phi=0.0):
    wl = jnp.asarray(wl)
    return sax.reciprocal({("o0", "o1"): jnp.exp(1j * phi) * jnp.ones_like(wl)})


MODELS = {"bdc": siepic.bidirectional_coupler, "wg": siepic.waveguide, "yb": siepic.y_branch,
          "gc": siepic.grating_coupler, "ps": _phase}


def netlist(corner=NOMINAL, L=LEN):
    th, w = corner
    ins, con, ports = {}, {}, {}

    def wg(name, length):
        ins[name] = {"component": "wg", "settings": {"length": float(length), "loss": LOSS, "width": w, "height": th}}
        return name

    def yb(name):
        ins[name] = {"component": "yb", "settings": {"thickness": th, "width": w}}

    def bdc(name):
        ins[name] = {"component": "bdc", "settings": {"thickness": th, "width": w}}

    def gc(name, port_name):
        ins[name] = {"component": "gc", "settings": {"thickness": th}}
        ports[port_name] = f"{name},o0"

    def switch(p):
        """Mach-Zehnder switch: in = {p}1,port_1; out = {p}2,port_3; heater phase on arm b."""
        bdc(p + "1"); bdc(p + "2"); wg(p + "a", L["mzi_arm"]); wg(p + "b", L["mzi_arm"]); ins[p + "h"] = "ps"
        con.update({f"{p}1,port_3": f"{p}a,o0", f"{p}a,o1": f"{p}2,port_1", f"{p}1,port_4": f"{p}h,o0",
                    f"{p}h,o1": f"{p}b,o0", f"{p}b,o1": f"{p}2,port_2"})
        ports[p + "_in2"] = f"{p}1,port_2"
        ports[p + "_dump"] = f"{p}2,port_4"
        return f"{p}1,port_1", f"{p}2,port_3"

    def loop_mirror(p):
        yb(p); wg(p + "l", L["loop"])
        con.update({f"{p},port_2": f"{p}l,o0", f"{p}l,o1": f"{p},port_3"})
        return f"{p},port_1"

    def michelson(p, in_port, arm_a, arm_b, sw_a, sw_b, bias, out_name, Lm):
        bdc(p)
        con[in_port] = f"{p},port_1"
        for arm, length, sw in (("a", arm_a, sw_a), ("b", arm_b, sw_b)):
            wg(f"{p}{arm}w", length)
            s_in, s_out = switch(sw)
            con[f"{p},{'port_3' if arm == 'a' else 'port_4'}"] = f"{p}{arm}w,o0"
            con[f"{p}{arm}w,o1"] = s_in
            if arm == "a":
                ins[bias] = "ps"
                con[s_out] = f"{bias},o0"
                con[f"{bias},o1"] = loop_mirror(f"{p}lm{arm}")
            else:
                con[s_out] = loop_mirror(f"{p}lm{arm}")
        wg(f"{p}out", Lm["x_out"]); gc(f"gc_{out_name}", out_name)
        con.update({f"{p},port_2": f"{p}out,o0", f"{p}out,o1": f"gc_{out_name},o1"})

    def and_branch(p, src, out_name):
        wg(p + "i", L[f"and_in_{p}"]); s_in, s_out = switch(p); wg(p + "o", L[f"and_out_{p}"]); gc("gc_" + out_name, out_name)
        con.update({src: f"{p}i,o0", f"{p}i,o1": s_in, s_out: f"{p}o,o0", f"{p}o,o1": f"gc_{out_name},o1"})

    def block(h, src, n1, n2, outs):
        """One half of the chip (layout.half_block): taps Sa, Sb; AND switches for n1, n2;
        Michelson XOR with arm switches n1 (arm a, with the bias heater) and n2 (arm b)."""
        sa, sb = f"Sa{h}", f"Sb{h}"
        yb(sa); yb(sb)
        wg(f"in{h}w", L[f"in{h}"]); wg(f"ab{h}w", L[f"ab{h}"]); wg(f"sx{h}w", L[f"sx{h}"])
        con.update({src: f"in{h}w,o0", f"in{h}w,o1": f"{sa},port_1", f"{sa},port_3": f"ab{h}w,o0",
                    f"ab{h}w,o1": f"{sb},port_1", f"{sb},port_3": f"sx{h}w,o0"})
        and_branch(f"{n1}_and", f"{sa},port_2", outs[0])
        and_branch(f"{n2}_and", f"{sb},port_2", outs[1])
        Lx = dict(L)
        Lx["x_out"] = L[f"x_out{h}"]
        michelson(f"X{h}", f"sx{h}w,o1", L[f"arm_a{h}"], L[f"arm_b{h}"], f"{n1}_x{h}", f"{n2}_x{h}", f"bias{h}", outs[2], Lx)

    gc("gc_in", "laser")
    yb("S1")
    wg("w_in", L["gc_s1"])
    con.update({"gc_in,o1": "w_in,o0", "w_in,o1": "S1,port_1"})
    block(1, "S1,port_2", "A", "B", ("c_a", "c_b", "xor1"))
    block(2, "S1,port_3", "C", "X", ("c_c", "c_d", "sum"))
    return {"instances": ins, "connections": con, "ports": ports}


class Chip:
    def __init__(self, corner=NOMINAL, L=LEN, lam=LAM):
        self.corner, self.lam = corner, lam
        self.circ, _ = sax.circuit(netlist=netlist(corner, L), models=MODELS)
        self.cal = None

    def run(self, **ph):
        """Output powers per watt of laser in the fibre. Keyword args: heater phases (rad), any of
        SWITCHES + BIASES, arrays broadcast together. Bias phases are round-trip values."""
        keys = SWITCHES + BIASES
        arrs = np.broadcast_arrays(*[np.atleast_1d(np.asarray(ph.get(k, 0.0), float)) for k in keys])
        shp, n = arrs[0].shape, arrs[0].size
        flat = [a.ravel() / (2.0 if k in BIASES else 1.0) for k, a in zip(keys, arrs)]   # bias heater is passed twice
        fields = {o: np.empty(n, complex) for o in OUTPUTS}
        CH = 6000                                                                       # batch size (memory)
        for i0 in range(0, n, CH):
            sl = slice(i0, min(n, i0 + CH))
            kw = {(k if k in BIASES else k + "h"): {"phi": jnp.asarray(v[sl])} for k, v in zip(keys, flat)}
            s = self.circ(wl=jnp.full(sl.stop - sl.start, self.lam * 1e6), **kw)
            for o in OUTPUTS:
                fields[o][sl] = np.asarray(s["laser", o])
        return {o: (np.abs(fields[o]) ** 2).reshape(shp) for o in OUTPUTS}, \
               {o: fields[o].reshape(shp) for o in ("xor1", "sum")}

    # ----------------------------------------------------------------- calibration (measurable)
    def calibrate(self, weights=(1.0, 1.0, 1.0, 0.0)):
        """Every step uses only grating-coupler outputs, as on a probe station.
        AND switches: sweep each heater on its own output. Arm switches: with the bias set for a
        bright output, an arm is OFF where the brightest output over the other arm's heater is
        dimmest. Bias: both arms ON, minimise the XOR output. Marking: set each AND switch's ON
        transmission so the four carry branches have the requested weights."""
        th = np.linspace(0, 2 * np.pi, 361, endpoint=False)
        cal = {}
        andk = {"A_and": "c_a", "B_and": "c_b", "C_and": "c_c", "X_and": "c_d"}
        full = {}
        base = {}
        for _ in range(2):                         # second pass: other branches parked OFF
            for sw, out in andk.items():
                kw = {k: v for k, v in base.items() if k != sw}
                p = self.run(**kw, **{sw: th})[0][out]
                cal[sw + "_off"] = float(th[np.argmin(p)])
                full[sw] = (float(th[np.argmax(p)]), float(p.max()), p)
            base = {k: cal[k + "_off"] for k in andk}
        for xor, (sa, sb), bias, out in (("x1", ("A_x1", "B_x1"), "bias1", "xor1"), ("x2", ("C_x2", "X_x2"), "bias2", "sum")):
            g = np.linspace(0, 2 * np.pi, 40, endpoint=False)
            TA, TB, PB = np.meshgrid(g, g, np.linspace(0, 2 * np.pi, 12, endpoint=False), indexing="ij")
            p = self.run(**{sa: TA, sb: TB, bias: PB})[0][out]
            ia, ib, ip = np.unravel_index(np.argmax(p), p.shape)
            on_a, on_b, bright = g[ia], g[ib], np.linspace(0, 2 * np.pi, 12, endpoint=False)[ip]
            f = np.linspace(0, 2 * np.pi, 181, endpoint=False)
            FA, FB = np.meshgrid(f, f, indexing="ij")
            p2 = self.run(**{sa: FA, sb: FB, bias: bright})[0][out]
            off_a = float(f[np.argmin(p2.max(axis=1))]); off_b = float(f[np.argmin(p2.max(axis=0))])
            # refine ON: maximise one-arm output with the other arm OFF
            on_a = float(f[np.argmax(self.run(**{sa: f, sb: off_b, bias: bright})[0][out])])
            on_b = float(f[np.argmax(self.run(**{sa: off_a, sb: f, bias: bright})[0][out])])
            ph = np.linspace(0, 2 * np.pi, 721, endpoint=False)
            d = self.run(**{sa: on_a, sb: on_b, bias: ph})[0][out]
            b0 = float(ph[np.argmin(d)])
            fine = b0 + np.linspace(-0.01, 0.01, 201)
            b0 = float(fine[np.argmin(self.run(**{sa: on_a, sb: on_b, bias: fine})[0][out])])
            cal.update({sa + "_on": on_a, sa + "_off": off_a, sb + "_on": on_b, sb + "_off": off_b, bias: b0})
        # marking: common scale = the weakest branch's full transmission / its weight
        w = dict(zip(andk, weights))
        scale = min(full[k][1] / w[k] for k in andk if w[k] > 0)
        for sw in andk:
            if w[sw] <= 0:
                cal[sw + "_on"] = cal[sw + "_off"]
                continue
            target = w[sw] * scale
            th_full = full[sw][0]
            # walk from OFF toward full ON along the monotonic side
            off = cal[sw + "_off"]
            path = off + np.linspace(0, ((th_full - off + np.pi) % (2 * np.pi)) - np.pi, 2001)
            pp = self.run(**{k: v for k, v in base.items() if k != sw}, **{sw: path})[0][andk[sw]]
            cal[sw + "_on"] = float(path[np.argmin(np.abs(pp - target))])
        self.cal, self.weights = cal, weights
        return self

    def phases(self, A, B, C, X, err=None):
        A, B, C, X = (np.asarray(v) for v in (A, B, C, X))
        c = self.cal
        sel = lambda k, bit: np.where(bit > 0, c[k + "_on"], c[k + "_off"])  # noqa: E731
        ph = {"A_and": sel("A_and", A), "B_and": sel("B_and", B), "C_and": sel("C_and", C), "X_and": sel("X_and", X),
              "A_x1": sel("A_x1", A), "B_x1": sel("B_x1", B), "C_x2": sel("C_x2", C), "X_x2": sel("X_x2", X),
              "bias1": c["bias1"] + np.zeros(np.shape(A)), "bias2": c["bias2"] + np.zeros(np.shape(A))}
        if err:
            for k, v in err.items():
                ph[k] = ph[k] + v
        return ph


# --------------------------------------------------------------------- detection and decisions
def det_sigma(i):
    i = np.asarray(i, float)
    var = 2 * P.QE * (i + DET["dark"]) + (DET["tia_pA"] * 1e-12) ** 2 + (i ** 2) * P.db2lin(DET["rin_dB"])
    return np.sqrt(var * DET["bw"])


TRUTH = [(a, b, c) for a in (0, 1) for b in (0, 1) for c in (0, 1)]


def evaluate_unit(chip, err=None, xor_thr=None):
    """All 8 inputs at 1 W in the fibre (outputs scale linearly with laser power). XOR1 is detected
    first; its comparator output X drives the X switches. err: heater phase errors (rad), arrays of
    shape (..., 1) broadcast against the 8 input combinations."""
    A, B, C = (np.array([t[k] for t in TRUTH]) for k in range(3))
    first, _ = chip.run(**chip.phases(A, B, C, np.zeros(8, int), err))
    x1 = first["xor1"]
    if xor_thr is None:
        xor_thr = 0.5 * (x1[..., (A ^ B) == 1].min(-1, keepdims=True) + x1[..., (A ^ B) == 0].max(-1, keepdims=True))
    X = (x1 > xor_thr).astype(int)
    out, _ = chip.run(**chip.phases(np.broadcast_to(A, X.shape), np.broadcast_to(B, X.shape), np.broadcast_to(C, X.shape), X, err))
    out["xor1"] = x1
    out["carry"] = out["c_a"] + out["c_b"] + out["c_c"] + out["c_d"]       # photodiodes in parallel
    return dict(A=A, B=B, C=C, X=X, u=out)


def margins(ev, p_laser=1e-3):
    """Decision quality for each output over the 8 inputs (last axis)."""
    A, B, C = ev["A"], ev["B"], ev["C"]
    want = {"xor1": A ^ B, "sum": A ^ B ^ C, "carry": (A & B) | (C & (A ^ B))}
    res = {}
    for k, t in want.items():
        i = DET["resp"] * p_laser * ev["u"][k]
        hi0, lo1 = i[..., t == 0].max(-1), i[..., t == 1].min(-1)
        q = (lo1 - hi0) / (det_sigma(hi0) + det_sigma(lo1))
        thr = (hi0 + lo1) / 2
        drift = np.minimum((lo1 - thr) / np.maximum(lo1, 1e-30), (thr - hi0) / np.maximum(hi0, 1e-30))
        res[k] = dict(threshold_uA=thr * 1e6, power_drift_tol_pct=np.minimum(drift, 10.0) * 100, contrast_dB=10 * np.log10(np.maximum(lo1, 1e-30) / np.maximum(hi0, 1e-30)),
                      margin_pct=(lo1 - hi0) / np.maximum(lo1, 1e-30) * 100, Q=q, correct=lo1 > hi0)
    return res


def scalar(m):
    return {k: {a: (bool(b) if isinstance(b, (np.bool_, bool)) or (hasattr(b, "dtype") and b.dtype == bool) else float(b))
                for a, b in v.items()} for k, v in m.items()}


def min_power(ev, q=7.0):
    lo, hi = 1e-9, 1e-1
    for _ in range(60):
        m = math.sqrt(lo * hi)
        ok = all(float(v["Q"]) >= q for v in margins(ev, m).values())
        lo, hi = (lo, m) if ok else (m, hi)
    return hi


def bias_tolerance(chip, which, p_laser=1e-3, limit_deg=90):
    """Largest |bias error| (degrees) for which every output still decides correctly with Q >= 7."""
    d = np.arange(0, limit_deg + 1)
    oks = []
    for sgn in (1, -1):
        ev = evaluate_unit(chip, err={which: sgn * np.radians(d)[:, None]})
        m = margins(ev, p_laser)
        oks.append(np.all([m[k]["Q"] >= 7 for k in m], axis=0))
    ok = oks[0] & oks[1]
    bad = np.where(~ok)[0]
    return float(d[bad[0] - 1]) if bad.size else float(limit_deg)


def cascade_failure(chip):
    """The light-only cascade of Fig. 15: XOR 1's output field added to a carry-in field at a
    combiner. For one arm open, XOR 1 leaves with opposite phase for 'only A' and 'only B'. For
    every carry-in phase phi and the best-matched amplitude, the sum output for (1,0,1) and
    (0,1,1) cannot both be dark."""
    c = chip.cal
    _, fa = chip.run(**chip.phases(np.array([1]), np.array([0]), np.array([0]), np.array([0])))
    _, fb = chip.run(**chip.phases(np.array([0]), np.array([1]), np.array([0]), np.array([0])))
    ea, eb = complex(fa["xor1"][0]), complex(fb["xor1"][0])
    phi = np.linspace(-np.pi, np.pi, 361)
    ec = abs(ea) * np.exp(1j * phi)             # carry-in field matched in amplitude
    t = 1 / math.sqrt(2)                        # Y-branch combiner, ideal
    p_a = np.abs(t * (ea + ec)) ** 2
    p_b = np.abs(t * (eb + ec)) ** 2
    p_one = np.abs(t * ea) ** 2                 # (1,0,0): should be 1
    return dict(phase_deg=np.degrees(phi).tolist(), A_only_plus_Cin=(p_a / p_one).tolist(),
                B_only_plus_Cin=(p_b / p_one).tolist(), relative_phase_deg=float(np.degrees(np.angle(eb / ea))),
                amplitude_ratio=float(abs(eb) / abs(ea)),
                best_worst_case=float(np.min(np.maximum(p_a, p_b) / p_one)))


def main():
    out = {"method": __doc__.strip().splitlines()[0], "detection": DET, "lengths_um": LEN}
    chips = {}
    for name, wts in (("majority", (1.0, 1.0, 1.0, 0.0)), ("paper_and_or", (1.0, 1.0, 0.5, 0.5))):
        ch = Chip().calibrate(wts)
        chips[name] = ch
        ev = evaluate_unit(ch)
        m = scalar(margins(ev))
        rows = [dict(A=int(ev["A"][k]), B=int(ev["B"][k]), Cin=int(ev["C"][k]), X=int(ev["X"][k]),
                     **{f"{o}_uA": float(ev["u"][o][k] * DET["resp"] * 1e-3 * 1e6) for o in ("xor1", "sum", "carry")}) for k in range(8)]
        out[name] = dict(weights=wts, calibration=ch.cal, truth_table=rows, margins=m, min_laser_uW=min_power(ev) * 1e6,
                         bias_tolerance_deg={"xor1": bias_tolerance(ch, "bias1"), "xor2": bias_tolerance(ch, "bias2")})
        print(name, {k: (round(v["margin_pct"], 1), round(v["contrast_dB"], 1), v["correct"]) for k, v in m.items()},
              "minP uW", round(out[name]["min_laser_uW"], 2), "tol", out[name]["bias_tolerance_deg"])
        for r in rows:
            print("  ", r)
    ch = chips["majority"]
    out["cascade_failure"] = cascade_failure(ch)
    print("cascade", {k: v for k, v in out["cascade_failure"].items() if not isinstance(v, list)})
    corners = []
    for cn in CORNERS:
        ck = Chip(cn).calibrate((1.0, 1.0, 1.0, 0.0))
        m = scalar(margins(evaluate_unit(ck)))
        fixed = Chip(cn); fixed.cal, fixed.weights = ch.cal, ch.weights
        mf = scalar(margins(evaluate_unit(fixed)))
        corners.append(dict(thickness_nm=cn[0], width_nm=cn[1],
                            recalibrated={k: dict(correct=v["correct"], margin_pct=v["margin_pct"], contrast_dB=v["contrast_dB"]) for k, v in m.items()},
                            nominal_heaters_correct=all(v["correct"] for v in mf.values())))
        print(cn, {k: round(v["margin_pct"], 1) for k, v in m.items()}, "nominal heaters ok:", corners[-1]["nominal_heaters_correct"])
    out["corners"] = corners
    with open(os.path.join(HERE, "full_adder_results.json"), "w") as f:
        json.dump(out, f, indent=1, default=float)
    return out


if __name__ == "__main__":
    main()
