#!/usr/bin/env python3
"""Export the single-pass XOR's time-domain network for the browser (../../encrypt/circuit.json), plus a
reference run that encrypt/engine.js is checked against (reference_run.json, not shipped)."""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from td import TDCircuit  # noqa: E402
from fast_xor_cal import FastXOR  # noqa: E402
from eyes import fast_setup, latency, drive, prbs  # noqa: E402

OUT = os.path.join(HERE, "..", "..", "encrypt")
fx = FastXOR()
cal = fx.calibrate()
tc = TDCircuit(extract=os.path.join(HERE, "..", "klayout", "fast_xor_extracted_netlist.json"))
base, sw = fast_setup(cal)
S = tc.S.tocoo()
lat = latency(tc, (base, sw))
truth = fx.truth()
circuit = dict(
    source="klayout/EBeam_heaters_atharvdubey_FastXOR (SiEPIC extraction), chip/speed/td.py",
    ports=tc.P, fibre=tc.fibre,
    S=[[int(r), int(c), float(v.real), float(v.imag)] for r, c, v in zip(S.row, S.col, S.data)],
    links=[dict(p=int(l["p"]), q=int(l["q"]), re=float(l["F"].real), im=float(l["F"].imag), D=float(l["D"]),
                heat=[[r, float(d)] for r, d in l["heat"]]) for l in tc.links],
    base={k: float(v) for k, v in base.items()},
    inputs={k: dict(role=v[0], on=float(v[1]), off=float(v[2])) for k, v in sw.items()},
    levels_W_per_W={k: v["xor"] for k, v in truth.items()},
    latency_ps=float(lat["first_ps"]),
)
with open(os.path.join(OUT, "circuit.json"), "w") as f:
    json.dump(circuit, f, separators=(",", ":"))

# reference: 64 bits at 100 Gb/s, ideal drive, two time steps
ref = {}
for dt in (0.02e-12, 0.05e-12):
    t = TDCircuit(extract=os.path.join(HERE, "..", "klayout", "fast_xor_extracted_netlist.json"), dt=dt)
    rate, nb = 100e9, 64
    lead = int(round(50e-12 / dt))
    n = lead + int(np.ceil((nb + 4) / rate / dt))
    a, b = prbs(7, nb, 1), prbs(7, nb, 50)
    per = dict(base)
    for key, bits in (("A", a), ("B", b)):
        role, on, off = sw[key]
        delta = ((on - off + np.pi) % (2 * np.pi)) - np.pi
        per[role] = off + delta * drive(bits, rate, dt, n, lead, None)
    e = t.run(n, per, t.fibre["laser1"], np.ones(n, complex), ["xor"])["xor"]
    ref[str(dt)] = dict(n=n, lead=lead, a=a.tolist(), b=b.tolist(), re=e.real.tolist(), im=e.imag.tolist())
json.dump(ref, open(os.path.join(HERE, "reference_run.json"), "w"))
print("exported", os.path.getsize(os.path.join(OUT, "circuit.json")), "bytes; latency", lat["first_ps"], "ps; levels", circuit["levels_W_per_W"])
