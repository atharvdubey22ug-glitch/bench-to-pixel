#!/usr/bin/env python3
"""
4-bit optical ALU: four copies of the hybrid optical full adder (builder.py) on one SiEPIC EBeam die,
in a 2 x 2 grid. Bit 0 is bottom-left, bit 1 bottom-right, bit 2 top-left, bit 3 top-right.

The bits are optically independent. They are chained electronically, the same way XOR 1 drives
XOR 2 inside each adder: bit i's carry photocurrent goes to a comparator, and its output drives
bit i+1's carry-in heaters (C_and, C_x2). The controller that does this also selects the operation:
  ADD   A + B           carry-in of bit 0 = 0, carries ripple
  SUB   A - B           B inverted in the heater driver, carry-in of bit 0 = 1 (two's complement)
  XOR   A xor B         read XOR 1 of every bit
  AND   A and B         carry-in of every bit = 0, read carry (the carry unit is majority(A, B, Cin))
  OR    A or B          carry-in of every bit = 1, read carry

Grating-coupler columns of the four bits sit on one 127 um grid (row pitch 1524 um = 12 x 127), so a
fibre array stepping down one column reaches every test circuit of bits 0 and 2 (and 1 and 3).

Outputs, in this folder (or the folder given as the first argument):
  EBeam_heaters_atharvdubey_ALU4.gds / .oas, .lyrdb   layout and SiEPIC verification report
  alu4_netlists/*.sp                                   SiEPIC SPICE netlist of each test circuit
  alu4_extracted_netlist.json                          SiEPIC net extraction (per bit) for ../alu4_sim.py
  alu4_build_report.json                               die, checks, pad map
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from builder import Builder, USER  # noqa: E402

OUT = sys.argv[1] if len(sys.argv) > 1 else HERE
TOP = f"EBeam_heaters_{USER}_ALU4"
PITCH = (1850.0, 1524.0)
GRID = {0: (0, 0), 1: (1, 0), 2: (0, 1), 3: (1, 1)}

b = Builder(TOP)
for bit, (gx, gy) in GRID.items():
    b.full_adder(bit, gx * PITCH[0], gy * PITCH[1],
                 opt_name=lambda h, bit=bit: f"ALU4_Bit{bit}_Half{h}",
                 pad_name=lambda h, k, bit=bit: f"ALU4_B{bit}_P{h}{k}",
                 pitch=PITCH, loopback=(bit == 0))
r = b.finish(OUT)

# every bit's drawn circuit must be identical (same components, lengths and roles, shifted)
loop_nets = {n for e in r["ext"] if str(e.get("role", "")).startswith("loopback") for n in e["pins"].values()}
for e in r["ext"]:                      # the alignment loopback (bit 0 only) is not part of the adder
    if e["kind"] == "Waveguide" and set(e["pins"].values()) <= loop_nets:
        e["role"] = "loopback_wg"
sig = {}
for bit, (gx, gy) in GRID.items():
    sig[bit] = [(e["kind"], e.get("role"), round(e.get("length_um") or 0, 3), e["center"][0] - gx * PITCH[0], e["center"][1] - gy * PITCH[1])
                for e in r["ext"] if e["bit"] == bit and not str(e.get("role", "")).startswith("loopback")]


def same(a, b, tol=0.05):          # same parts at the same place (0.05 um: coordinate rounding)
    if len(a) != len(b):
        return False
    left = list(b)
    for p in a:
        k = next((i for i, q in enumerate(left) if q[:3] == p[:3] and abs(q[3] - p[3]) < tol and abs(q[4] - p[4]) < tol), None)
        if k is None:
            return False
        left.pop(k)
    return True


assert all(same(sig[0], sig[k]) for k in sig), "bits differ"

os.makedirs(os.path.join(OUT, "alu4_netlists"), exist_ok=True)
for opt, sp in r["spice"].items():
    with open(os.path.join(OUT, "alu4_netlists", opt.split(f"{USER}_")[1] + ".sp"), "w") as f:
        f.write(sp)
with open(os.path.join(OUT, "alu4_extracted_netlist.json"), "w") as f:
    json.dump(dict(bits=list(GRID), components=r["ext"]), f, indent=1)
report = dict(top_cell=TOP, die_um=r["die"], bits=len(GRID), grid=GRID, siepic_verification_errors=r["n_err"],
              components=len(r["ext"]), components_per_bit=len(sig[0]), bits_identical=True,
              test_circuits=r["spice"] and sorted(r["spice"]), waveguide_crossings=r["crossings"],
              dangling_optical_pins=r["dangling"], pads=len([p for blk in b.blocks for p in blk["pads"]]),
              heater_pads=[dict(bit=blk["bit"], **hp) for blk in b.blocks for hp in blk["heaters"]])
with open(os.path.join(OUT, "alu4_build_report.json"), "w") as f:
    json.dump(report, f, indent=1)
print("die", report["die_um"], "| SiEPIC verification errors:", r["n_err"], "| components", len(r["ext"]),
      f"({len(sig[0])} per bit, identical)", "| crossings", len(r["crossings"]), "| dangling optical pins",
      len(r["dangling"]), "| pads", report["pads"], "| test circuits", len(r["spice"]))
