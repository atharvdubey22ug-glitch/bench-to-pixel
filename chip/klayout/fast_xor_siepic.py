#!/usr/bin/env python3
"""
Single-pass optical XOR for high-speed data: the paper's two-beam XOR with the beams recombined at a
second coupler instead of being folded back by loop mirrors, so each input modulator is crossed once.

    laser -> Y-branch -> arm A: switch A (input A) -> bias heater A --\\
                      -> arm B: switch B (input B) -> bias heater B --- broadband coupler -> XOR   (y = 254)
                                                                                         -> monitor (y = 0)

Arm B is the mirror image of arm A, so the arms are equal in length. Both coupler outputs are routed
to grating couplers in one column (Phot1x design-for-test: laser at y = 127, one detector above and
one below). On this EBeam heater process the "modulators" are the same thermo-optic switches as in
the full adder; the high-speed study (../speed/) treats them as fast phase modulators of the same
geometry, i.e. this circuit built in a process that has them.
Outputs: EBeam_heaters_atharvdubey_FastXOR.gds/.oas/.lyrdb, fast_xor_extracted_netlist.json,
fast_xor_build_report.json.
"""
import json
import os
import sys

import pya
from pya import DPoint, DCplxTrans

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from builder import Builder, USER, D, HEATER, P  # noqa: E402

OUT = sys.argv[1] if len(sys.argv) > 1 else HERE
TOP = f"EBeam_heaters_{USER}_FastXOR"
XC = 700.0                    # combiner coupler centre (x), at y = 127

b = Builder(TOP)
b.heaters_per_bit = 4
b.bit, b.pre = 0, ""
b.SHIFT = DCplxTrans(1, 0, False, 120.0, 320.0)
ID = DCplxTrans()
M = DCplxTrans(1, 0, True, 0, 254.0)          # mirror about y = 127: arm B

for r in (0.0, 127.0, 254.0):
    b.place(b.C_GC, ID, 0, r)
b.place(b.C_Y, ID, 60.0, 127.0)
b.wg("laser_in", [DPoint(0, 127), DPoint(52.6, 127)])
b.place(b.C_BDC, ID, XC, 127.0)


def arm(T, mod, bias, side):
    """One arm in arm-A coordinates (T = identity for A, mirror for B)."""
    b.wg(f"arm_{side}", [P(T, 67.4, 129.75), P(T, 90, 129.75), P(T, 90, 200), P(T, 150, 200)])
    sw = b.switch(mod, T * DCplxTrans(1, 0, True, 150 + 35.45, 200 + 2.35))      # heater on the outer side
    xo = (T.inverted() * sw["out"]).x
    xh = xo + 30.0
    b.wg(f"arm_{side}", [P(T, xo, 200), P(T, xh, 200)])
    hb = b.place(b.C_HEAT, T, xh, 200)
    e1, e2 = b.pinp(hb, "elec1"), b.pinp(hb, "elec2")
    b.heaters[bias] = sorted([(e1.x, e1.y), (e2.x, e2.y)])
    b.lengths[f"arm_{side}"] += HEATER
    xw = XC - 35.45 - 25.0
    b.wg(f"arm_{side}", [P(T, xh + HEATER, 200), P(T, xw, 200), P(T, xw, 129.35), P(T, XC - 35.45, 129.35)])
    # pads (bottom edge y = 380 in arm-A coordinates) and the chained heater wiring
    y_pad, pads_x = 380.0, (200.0, 400.0, 600.0)
    names = []
    for k, x in enumerate(pads_x):
        b.place(b.C_PAD, T, x, y_pad + 50.0)
        names.append(f"FastXOR_P{side}{k}")
        b.label(f"elec_{USER}_{names[-1]}", T * DPoint(x, y_pad + 50.0))
    (L1, R1), (L2, R2) = [sorted([T.inverted() * DPoint(*p) for p in b.heaters[n]], key=lambda q: q.x) for n in (mod, bias)]

    def W(pts):
        b.wire([T * DPoint(*p) if isinstance(p, tuple) else T * p for p in pts])

    W([L1, (L1.x, 360.0), (pads_x[0], 360.0), (pads_x[0], y_pad)])
    W([R1, (R1.x, 340.0), (L2.x, 340.0)])
    W([L2, (L2.x, 340.0)])
    W([(pads_x[1], 340.0), (pads_x[1], y_pad)])
    W([R2, (R2.x, 360.0), (pads_x[2], 360.0), (pads_x[2], y_pad)])
    return dict(pads=names, heaters=[dict(name=mod, between=names[:2]), dict(name=bias, between=names[1:])])


pa = arm(ID, "A_mod", "biasA", "A")
pb = arm(M, "B_mod", "biasB", "B")
b.wg("xor_out", [DPoint(XC + 35.3, 129.35), DPoint(XC + 55, 129.35), DPoint(XC + 55, 300), DPoint(40, 300), DPoint(40, 254), DPoint(0, 254)])
b.wg("mon_out", [DPoint(XC + 35.3, 124.65), DPoint(XC + 55, 124.65), DPoint(XC + 55, -46), DPoint(40, -46), DPoint(40, 0), DPoint(0, 0)])
opt = f"opt_in_TE_1550_device_{USER}_FastXOR"
b.label(opt, DPoint(0, 127))
b.opt_in.append(opt)
for r, role in ((254.0, "xor"), (127.0, "laser1"), (0.0, "mon")):
    b.gc_roles[(0, role)] = b.SHIFT * DPoint(0, r)
b.bits.append((0, pya.DBox(-1e4, -1e4, 1e4, 1e4), "", b.SHIFT))
b.blocks += [dict(bit=0, block="A", **pa), dict(bit=0, block="B", **pb)]

r = b.finish(OUT, dft_at=(60.0, 860.0))
with open(os.path.join(OUT, "fast_xor_extracted_netlist.json"), "w") as f:
    json.dump(dict(components=r["ext"]), f, indent=1)
report = dict(top_cell=TOP, die_um=r["die"], siepic_verification_errors=r["n_err"], components=len(r["ext"]),
              waveguide_crossings=r["crossings"], dangling_optical_pins=r["dangling"],
              lengths_um={k: round(v, 3) for k, v in b.lengths.items()},
              heater_pads=pa["heaters"] + pb["heaters"])
with open(os.path.join(OUT, "fast_xor_build_report.json"), "w") as f:
    json.dump(report, f, indent=1)
print("die", report["die_um"], "| SiEPIC verification errors:", r["n_err"], "| components", len(r["ext"]),
      "| crossings", len(r["crossings"]), "| dangling", len(r["dangling"]),
      "| arm A", report["lengths_um"]["arm_A"], "arm B", report["lengths_um"]["arm_B"])
