#!/usr/bin/env python3
"""
Phase-encoded 8-bit parity: one interferometer computes b1 XOR b2 XOR ... XOR b8 in a single pass.

    laser -> Y-branch -> signal arm:    S1  S2  S3  S4  S5  S6  S7  S8  bias  --\\
                      -> reference arm: plain waveguide, the mirror image       --- broadband coupler -> PARITY     (y = 254)
                                                                                                    -> NOT PARITY (y = 0)

Each input bit b_k drives one phase section of the signal arm: 0 -> phase 0, 1 -> phase pi. Phases
add along the arm, so the light leaving it carries phase pi * (b1 + ... + b8), i.e. pi * parity. The
coupler turns that phase into intensity against the reference arm. The bits are encoded in phase, not
in amplitude, which is why the sections cascade without detection in between (the amplitude-encoded
XORs of the full adder cannot be chained: see full_adder_sim.cascade_failure()).

The reference arm is the mirror image of the signal arm (same bends, same length), so the two arms are
balanced and the result does not depend on wavelength. On this EBeam heater process the phase sections
are the TiW heaters (wg_heater, 120 um); the speed study (../speed/parity_speed.py) treats them as fast
phase modulators of the same placement. The nine heaters are chained on ten pads (heater k between
pads k and k+1), which still sets each heater's power independently.

Outputs: EBeam_heaters_atharvdubey_Parity8.gds/.oas/.lyrdb, parity_extracted_netlist.json,
parity_build_report.json.
"""
import json
import os
import sys

import pya
from pya import DPoint, DCplxTrans

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from builder import Builder, USER, HEATER, P  # noqa: E402

OUT = sys.argv[1] if len(sys.argv) > 1 else HERE
TOP = f"EBeam_heaters_{USER}_Parity8"
N = 8                          # data sections
PITCH = 150.0                  # heater pitch along the arm (= pad pitch)
X0 = 160.0                     # start of the first section
Y_ARM = 200.0                  # signal arm (reference arm: mirror image about y = 127, at y = 54)
NAMES = [f"S{k + 1}" for k in range(N)] + ["bias"]
XS = [X0 + PITCH * k for k in range(len(NAMES))]          # heater starts
XC = XS[-1] + HEATER + 120.0   # combiner coupler centre
Y_PAD = 380.0                  # pad bottom edge

b = Builder(TOP)
b.heaters_per_bit = len(NAMES)
b.bit, b.pre = 0, ""
b.SHIFT = DCplxTrans(1, 0, False, 120.0, 130.0)
ID = DCplxTrans()
M = DCplxTrans(1, 0, True, 0, 254.0)          # mirror about y = 127

for r in (0.0, 127.0, 254.0):
    b.place(b.C_GC, ID, 0, r)
b.place(b.C_Y, ID, 60.0, 127.0)
b.wg("laser_in", [DPoint(0, 127), DPoint(52.6, 127)])
b.place(b.C_BDC, ID, XC, 127.0)

xw = XC - 35.45 - 25.0
head = [(67.4, 129.75), (90, 129.75), (90, Y_ARM), (XS[0], Y_ARM)]
tail = [(XS[-1] + HEATER, Y_ARM), (xw, Y_ARM), (xw, 129.35), (XC - 35.45, 129.35)]

# signal arm: lead-in, nine heaters with short waveguides between them, lead-out
b.wg("arm_sig", [P(ID, *p) for p in head])
for k, nm in enumerate(NAMES):
    h = b.place(b.C_HEAT, ID, XS[k], Y_ARM)
    e1, e2 = b.pinp(h, "elec1"), b.pinp(h, "elec2")
    b.heaters[nm] = sorted([(e1.x, e1.y), (e2.x, e2.y)])
    b.lengths["arm_sig"] += HEATER
    if k + 1 < len(NAMES):
        b.wg("arm_sig", [DPoint(XS[k] + HEATER, Y_ARM), DPoint(XS[k + 1], Y_ARM)])
b.wg("arm_sig", [P(ID, *p) for p in tail])
# reference arm: the same path, mirrored, as one plain waveguide
b.wg("arm_ref", [P(M, *p) for p in head + tail])

# pads and the chained heater wiring: pad k sits between heater k-1 and heater k
pads = []
for k in range(len(NAMES) + 1):
    x = XS[0] - 15.0 + PITCH * k
    b.place(b.C_PAD, ID, x, Y_PAD + 50.0)
    pads.append(f"Parity_P{k}")
    b.label(f"elec_{USER}_{pads[-1]}", DPoint(x, Y_PAD + 50.0))
    ends = []
    if k > 0:
        ends.append(DPoint(*b.heaters[NAMES[k - 1]][1]))      # right contact of heater k-1
    if k < len(NAMES):
        ends.append(DPoint(*b.heaters[NAMES[k]][0]))          # left contact of heater k
    yj = 330.0
    for e in ends:
        b.wire([e, DPoint(e.x, yj)])
    if len(ends) == 2:
        b.wire([DPoint(ends[0].x, yj), DPoint(ends[1].x, yj)])
    else:
        b.wire([DPoint(ends[0].x, yj), DPoint(x, yj)])
    b.wire([DPoint(x, yj), DPoint(x, Y_PAD)])

b.wg("par_out", [DPoint(XC + 35.3, 129.35), DPoint(XC + 55, 129.35), DPoint(XC + 55, 300), DPoint(40, 300), DPoint(40, 254), DPoint(0, 254)])
b.wg("npar_out", [DPoint(XC + 35.3, 124.65), DPoint(XC + 55, 124.65), DPoint(XC + 55, -46), DPoint(40, -46), DPoint(40, 0), DPoint(0, 0)])
opt = f"opt_in_TE_1550_device_{USER}_Parity8"
b.label(opt, DPoint(0, 127))
b.opt_in.append(opt)
for r, role in ((254.0, "parity"), (127.0, "laser1"), (0.0, "nparity")):
    b.gc_roles[(0, role)] = b.SHIFT * DPoint(0, r)
b.bits.append((0, pya.DBox(-1e4, -1e4, 1e4, 1e4), "", b.SHIFT))
heater_pads = [dict(name=nm, between=[pads[k], pads[k + 1]]) for k, nm in enumerate(NAMES)]
b.blocks.append(dict(bit=0, block="parity", pads=pads, heaters=heater_pads))

r = b.finish(OUT, dft_at=(60.0, 560.0))
with open(os.path.join(OUT, "parity_extracted_netlist.json"), "w") as f:
    json.dump(dict(components=r["ext"]), f, indent=1)
report = dict(top_cell=TOP, die_um=r["die"], siepic_verification_errors=r["n_err"], components=len(r["ext"]),
              waveguide_crossings=r["crossings"], dangling_optical_pins=r["dangling"],
              lengths_um={k: round(v, 3) for k, v in b.lengths.items()},
              section_pitch_um=PITCH, heater_pads=heater_pads)
with open(os.path.join(OUT, "parity_build_report.json"), "w") as f:
    json.dump(report, f, indent=1)
print("die", report["die_um"], "| SiEPIC verification errors:", r["n_err"], "| components", len(r["ext"]),
      "| crossings", len(r["crossings"]), "| dangling", len(r["dangling"]),
      "| signal arm", report["lengths_um"]["arm_sig"], "reference arm", report["lengths_um"]["arm_ref"])
