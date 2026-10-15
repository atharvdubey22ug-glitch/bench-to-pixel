#!/usr/bin/env python3
"""Export the 4-bit ALU's simulated light levels for the ALU page (../alu/levels.json).

For every bit, every die of alu4_sim.py, and all 16 states of the bit's inputs (A, B, Cin and the
re-applied XOR 1 decision X), the outputs XOR 1, SUM and CARRY (sum of the four carry photodiodes)
as photocurrent per watt of laser per bit, from that bit's circuit in the SiEPIC extraction of the
ALU layout. Thresholds are the comparator thresholds alu4_sim.py set for each die."""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import full_adder_sim as F  # noqa: E402
import alu4_sim as A  # noqa: E402

res = json.load(open(os.path.join(HERE, "alu4_results.json")))
nominal_cal = json.load(open(os.path.join(HERE, "full_adder_results.json")))["majority"]["calibration"]
stressed_cal = res["calibration_stressed"]
dies = {
    "nominal": dict(label="Nominal die", corners=[F.NOMINAL] * 4, cal=[nominal_cal] * 4),
    "stressed": dict(label="Stressed die, each bit calibrated", corners=[A.STRESSED[i] for i in range(4)], cal=stressed_cal),
    "shared_heaters": dict(label="Stressed die, nominal heater settings, own thresholds", corners=[A.STRESSED[i] for i in range(4)], cal=[nominal_cal] * 4),
    "no_calibration": dict(label="Stressed die, no calibration", corners=[A.STRESSED[i] for i in range(4)], cal=[nominal_cal] * 4),
}
st = [(a, b, c, x) for a in (0, 1) for b in (0, 1) for c in (0, 1) for x in (0, 1)]
Av, Bv, Cv, Xv = (np.array([s[k] for s in st]) for k in range(4))
out = dict(source="chip/alu4_sim.py on klayout/EBeam_heaters_atharvdubey_ALU4 (SiEPIC extraction)",
           state_order="index = 8*A + 4*B + 2*Cin + X", detection=F.DET, laser_per_bit_W=A.P_BIT, dies={})
for name, d in dies.items():
    bits = []
    for i in range(4):
        ch = F.Chip(d["corners"][i], extract=A.EXTRACT, bit=i)
        ch.cal = d["cal"][i]
        u, _ = ch.run(**ch.phases(Av, Bv, Cv, Xv))
        thr = res[name]["thresholds_uA"][i]
        bits.append(dict(corner=dict(thickness_nm=d["corners"][i][0], width_nm=d["corners"][i][1]),
                         xor1=[float(v) for v in u["xor1"]], sum=[float(v) for v in u["sum"]],
                         carry=[float(v) for v in u["c_a"] + u["c_b"] + u["c_c"] + u["c_d"]],
                         thr={k: v * 1e-6 / A.P_BIT for k, v in thr.items()}))      # threshold per watt of laser
    out["dies"][name] = dict(label=d["label"], bits=bits, python_wrong=res[name]["total_wrong"],
                             python_ops=res[name]["total_operations"])
    print(name, "done")
with open(os.path.join(HERE, "..", "alu", "levels.json"), "w") as f:
    json.dump(out, f, separators=(",", ":"))
