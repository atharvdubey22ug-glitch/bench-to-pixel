#!/usr/bin/env python3
"""
4-bit optical ALU: circuit simulation of klayout/EBeam_heaters_atharvdubey_ALU4 (four hybrid full
adders on one die), read from the SiEPIC net extraction of that layout (klayout/alu4_extracted_netlist.json).

Each bit is simulated from its own extracted circuit (one SAX scattering network per bit, PDK
S-matrices, drawn waveguide lengths) and calibrated with grating-coupler measurements only, as in
full_adder_sim.py. The electronic controller is modelled as it would be built:
  * comparator thresholds are fixed per bit after calibration (midpoints of the measured levels);
  * every decision sees detector noise (shot, TIA, laser RIN at 1 MHz bandwidth);
  * the carry ripples: bit i's carry comparator drives bit i+1's carry-in heaters;
  * the operation sets B's inversion and the carry-ins:
      ADD  A + B + cin            SUB  A + not(B) + 1  (= A - B, carry out = no borrow)
      AND  carry with every carry-in 0 (majority(A, B, 0) = A and B)
      OR   carry with every carry-in 1 (majority(A, B, 1) = A or B)
      XOR  XOR 1 of every bit
Each bit gets 1 mW of laser (a 4 mW laser split eight ways, 0.5 mW per test circuit).

Dies simulated
  nominal     every bit at 220 nm silicon / 500 nm waveguides
  stressed    each bit at a different process corner (+-10 nm thickness, +-20 nm width), far more
              within-die variation than a real 3.7 mm die has; each bit calibrated on its own
  shared_heaters  the stressed die with the nominal die's heater settings, but each bit's comparator
                  thresholds set from its own 8 measured states
  no_calibration  the stressed die with the nominal die's heater settings and thresholds
"""
import json
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import full_adder_sim as F  # noqa: E402

EXTRACT = os.path.join(HERE, "klayout", "alu4_extracted_netlist.json")
P_BIT = 1e-3
NBITS = 4
STRESSED = {0: (210.0, 480.0), 1: (230.0, 520.0), 2: (210.0, 520.0), 3: (230.0, 480.0)}
OPS = ("ADD", "SUB", "AND", "OR", "XOR")


class Bit:
    def __init__(self, bit, corner, cal_from=None, own_thresholds=False):
        """cal_from: reuse another bit's heater settings (and, unless own_thresholds, its thresholds).
        Thresholds are the midpoints of the bit's measured levels over its 8 input states."""
        self.bit, self.corner = bit, corner
        self.chip = F.Chip(corner, extract=EXTRACT, bit=bit)
        if cal_from is None:
            self.chip.calibrate((1.0, 1.0, 1.0, 0.0))
        else:
            self.chip.cal, self.chip.weights, self.thr = cal_from.chip.cal, cal_from.chip.weights, cal_from.thr
        if cal_from is None or own_thresholds:
            ev = F.evaluate_unit(self.chip)
            m = F.margins(ev, P_BIT)
            self.thr = {k: float(m[k]["threshold_uA"]) * 1e-6 for k in ("xor1", "sum", "carry")}

    def step(self, a, b, c, rng):
        """One bit, vectorised. Returns decisions and currents (A) for xor1, sum, carry."""
        z = np.zeros_like(a)
        u1, _ = self.chip.run(**self.chip.phases(a, b, c, z))
        i_x = F.DET["resp"] * P_BIT * u1["xor1"]
        i_x = i_x + rng.standard_normal(i_x.shape) * F.det_sigma(i_x)
        x = (i_x > self.thr["xor1"]).astype(int)
        u2, _ = self.chip.run(**self.chip.phases(a, b, c, x))
        i_s = F.DET["resp"] * P_BIT * u2["sum"]
        i_c = F.DET["resp"] * P_BIT * (u2["c_a"] + u2["c_b"] + u2["c_c"] + u2["c_d"])
        i_s = i_s + rng.standard_normal(i_s.shape) * F.det_sigma(i_s)
        i_c = i_c + rng.standard_normal(i_c.shape) * F.det_sigma(i_c)
        return dict(xor1=x, sum=(i_s > self.thr["sum"]).astype(int), carry=(i_c > self.thr["carry"]).astype(int),
                    I=dict(xor1=i_x, sum=i_s, carry=i_c))


def alu(bits, op, A, B, cin, rng, log):
    """Run one operation on arrays of 4-bit operands. log[(bit, output)] collects (current, wanted) pairs."""
    res = np.zeros_like(A)
    c = cin.copy()
    for i, bt in enumerate(bits):
        a = (A >> i) & 1
        b = (B >> i) & 1
        if op == "SUB":
            b = 1 - b
        if op in ("AND", "OR"):
            c = np.full_like(a, 1 if op == "OR" else 0)
        out = bt.step(a, b, c, rng)
        want = {"xor1": a ^ b, "sum": a ^ b ^ c, "carry": (a & b) | (c & (a ^ b))}
        for k in ("xor1", "sum", "carry"):
            log.setdefault((i, k), []).append((out["I"][k], want[k]))
        bit_out = {"ADD": out["sum"], "SUB": out["sum"], "AND": out["carry"], "OR": out["carry"], "XOR": out["xor1"]}[op]
        res |= bit_out << i
        c = out["carry"]
    return res, c


def reference(op, A, B, cin):
    if op == "ADD":
        s = A + B + cin
        return s & 15, s >> 4
    if op == "SUB":
        s = A + (15 - B) + 1
        return s & 15, s >> 4
    return {"AND": A & B, "OR": A | B, "XOR": A ^ B}[op], None


def run_die(bits, seed=0):
    rng = np.random.default_rng(seed)
    AA, BB = np.meshgrid(np.arange(16), np.arange(16), indexing="ij")
    A0, B0 = AA.ravel(), BB.ravel()
    results, log = {}, {}
    for op in OPS:
        cins = (0, 1) if op == "ADD" else (1 if op == "SUB" else 0,)
        wrong, n = 0, 0
        for ci in cins:
            cin = np.full_like(A0, ci)
            got, cout = alu(bits, op, A0, B0, cin, rng, log)
            ref, rcout = reference(op, A0, B0, cin)
            bad = got != ref
            if rcout is not None:
                bad |= cout != rcout
            wrong += int(bad.sum())
            n += A0.size
        results[op] = dict(operations=n, wrong=wrong)
    # analog quality of every decision made: distance to the threshold in units of the noise
    quality = {}
    for (i, k), rows in sorted(log.items()):
        I = np.concatenate([r[0] for r in rows])
        w = np.concatenate([r[1] for r in rows])
        thr = bits[i].thr[k]
        lo1, hi0 = I[w == 1].min(), I[w == 0].max()
        q = (lo1 - hi0) / (F.det_sigma(lo1) + F.det_sigma(hi0))
        quality[f"bit{i}_{k}"] = dict(decisions=int(I.size), threshold_uA=thr * 1e6, lowest_one_uA=float(lo1 * 1e6),
                                      highest_zero_uA=float(hi0 * 1e6), Q=float(q),
                                      margin_pct=float((lo1 - hi0) / lo1 * 100))
    return results, quality, log


def eye_data(log, bits):
    out = {}
    for (i, k), rows in log.items():
        I = np.concatenate([r[0] for r in rows]) * 1e6
        w = np.concatenate([r[1] for r in rows])
        out[f"bit{i}_{k}"] = dict(ones=np.round(np.sort(I[w == 1]), 3).tolist(), zeros=np.round(np.sort(I[w == 0]), 3).tolist(),
                                  threshold=bits[i].thr[k] * 1e6)
    return out


def main():
    t0 = time.time()
    out = {"method": __doc__.strip().splitlines()[0], "netlist": os.path.relpath(EXTRACT, HERE), "laser_per_bit_mW": P_BIT * 1e3,
           "detection": F.DET, "stressed_corners": {str(k): dict(thickness_nm=v[0], width_nm=v[1]) for k, v in STRESSED.items()}}
    nominal_bit = Bit(0, F.NOMINAL)
    nominal = [nominal_bit] + [Bit(i, F.NOMINAL, cal_from=nominal_bit) for i in range(1, NBITS)]   # identical bits
    stressed = [Bit(i, STRESSED[i]) for i in range(NBITS)]
    shared = [Bit(i, STRESSED[i], cal_from=nominal_bit, own_thresholds=True) for i in range(NBITS)]
    nocal = [Bit(i, STRESSED[i], cal_from=nominal_bit) for i in range(NBITS)]
    eyes = {}
    for name, bits in (("nominal", nominal), ("stressed", stressed), ("shared_heaters", shared), ("no_calibration", nocal)):
        res, qual, log = run_die(bits, seed=1)
        minp = []
        for bt in bits:
            ev = F.evaluate_unit(bt.chip)
            minp.append(F.min_power(ev) * 1e6)
        out[name] = dict(operations=res, total_wrong=sum(v["wrong"] for v in res.values()),
                         total_operations=sum(v["operations"] for v in res.values()),
                         worst_Q=min(v["Q"] for v in qual.values()), worst_margin_pct=min(v["margin_pct"] for v in qual.values()),
                         decisions=qual, min_laser_per_bit_uW=minp, min_laser_total_uW=4 * max(minp),
                         thresholds_uA=[{k: v * 1e6 for k, v in bt.thr.items()} for bt in bits])
        eyes[name] = eye_data(log, bits)
        print(name, {k: v["wrong"] for k, v in res.items()}, "of", out[name]["total_operations"],
              "| worst Q", round(out[name]["worst_Q"], 1), "| worst margin %", round(out[name]["worst_margin_pct"], 1),
              "| min laser total uW", round(out[name]["min_laser_total_uW"], 1), f"| {time.time() - t0:.0f}s")
    out["calibration_stressed"] = [bt.chip.cal for bt in stressed]
    with open(os.path.join(HERE, "alu4_results.json"), "w") as f:
        json.dump(out, f, indent=1, default=float)
    with open(os.path.join(HERE, "alu4_eyes.json"), "w") as f:
        json.dump(eyes, f)
    return out


if __name__ == "__main__":
    main()
