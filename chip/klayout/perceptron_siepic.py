#!/usr/bin/env python3
"""
Optical binary neuron: sixteen inputs, one pass of light, one weighted sum.

    laser -> Y0 -> reference arm (trim heater) -------------------------------------------\\
              \\-> 1x16 Y-branch tree -> 16 arms, one phase section each -> 16x1 tree ----- coupler -> OUT+ (y = +127)
                                                                                                   -> OUT- (y = -127)

Each arm carries one input: its phase section is set to 0 or pi by (input bit XOR weight bit), so the
arm's light arrives as +a or -a. The combiner tree adds the sixteen fields coherently: its output is
a * sum_k w_k x_k with w, x in {-1, +1}. The coupler mixes that sum with the reference; the difference
of the two output photocurrents is proportional to the signed sum (balanced homodyne detection), and a
comparator on it is the neuron's threshold. With w and x in {-1, +1} the sum is 16 - 2 * (number of
pixels that differ from the weight pattern): an XNOR-popcount neuron, as in binarized neural networks.

Geometry. The two trees are mirror images and every arm is a straight line between them, so all
sixteen paths have the same length. The reference arm's depth is chosen so its length matches the
signal path. The heaters form a staircase (outer arms further right) so each heater's wires reach the
pad row straight, crossing other arms only where they have no heater. Upper-half heaters are chained
on 9 pads at the top, lower-half heaters on 9 pads at the bottom (heater u between pads u and u+1);
the reference trim heater has its own 2 pads. On this EBeam process the phase sections are TiW
heaters; ../speed/perceptron_speed.py treats them as fast phase modulators of the same placement.

Outputs: EBeam_heaters_atharvdubey_Perceptron16.gds/.oas/.lyrdb, perceptron_extracted_netlist.json,
perceptron_build_report.json.
"""
import json
import os
import sys

import pya
from pya import DPoint, DCplxTrans

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from builder import Builder, USER, HEATER, P as PT  # noqa: E402

OUT = sys.argv[1] if len(sys.argv) > 1 else HERE
TOP = f"EBeam_heaters_{USER}_Perceptron16"
PITCH = 100.0                    # arm pitch
Y0P = 2.75                       # Y0's upper port
TREE_UP = 70.0                   # trees raised above Y0, so the signal path is as long as the reference
YT = Y0P + TREE_UP               # centre line of the trees
X1, DX, A = 100.0, 50.0, 12.0    # first split Y-branch, spacing between tree levels, lead before each turn
XLT = X1 + 3 * DX + 7.4 + A      # x where the leaves turn into the arms
X0, DS = 420.0, 150.0            # staircase: heater of step u starts at X0 + u * DS
XMID = 940.0                     # mirror axis between the two trees
XC = 2 * XMID - X1 + 7.4 + 35.45 + 60.0   # output coupler centre
YB = Y0P - 2.35                  # coupler centre (upper-left port at y = Y0P)
Y_RT = YT + 7.5 * PITCH + 87.0   # OUT+ route, above the top arm's wiring
Y_PAD_TOP = Y_RT + 40.0          # top pad row, bottom edge
Y_REF = float(sys.argv[2]) if len(sys.argv) > 2 else -798.938   # reference depth (length-matched, see report)
H = [4 * PITCH, 2 * PITCH, PITCH, PITCH / 2]

b = Builder(TOP)
b.heaters_per_bit = 17
b.bit, b.pre = 0, ""
b.SHIFT = DCplxTrans(1, 0, False, 120.0, 1040.0)
ID = DCplxTrans()
MX = DCplxTrans(1, 180, True, 2 * XMID, 0)        # x -> 2 XMID - x: the combiner tree
MY = DCplxTrans(1, 0, True, 0, 2 * YT)             # y -> 2 YT - y: lower half of the staircase


def arm_index(u, upper):
    """Leaf number k (0 = bottom, 15 = top) of staircase step u in the upper or lower half."""
    return 8 + u if upper else 7 - u


def leaf_y(k):
    return YT + (k - 7.5) * PITCH


# ---------------------------------------------------------------- GCs, Y0, the two trees
for r in (-127.0, 0.0, 127.0):
    b.place(b.C_GC, ID, 0, r)
b.wg("laser_in", [DPoint(0, 0), DPoint(52.6, 0)])
b.place(b.C_Y, ID, 60.0, 0.0)
b.wg("sig_in", [DPoint(67.4, Y0P), DPoint(75.0, Y0P), DPoint(75.0, YT), DPoint(X1 - 7.4, YT)])


def tree(T, key):
    """Y-branch tree, root at (X1, YT) in split coordinates; returns the leaf port of each leaf k
    (where the arm's turn begins), in split coordinates."""
    nodes = [YT]
    leaves = {}
    for lv in range(4):
        x = X1 + lv * DX
        nxt = []
        for c in nodes:
            b.place(b.C_Y, T, x, c)
            for s in (1, -1):
                y = c + s * H[lv]
                if lv < 3:
                    b.wg(f"{key}{lv}", [PT(T, x + 7.4, c + s * 2.75), PT(T, x + 7.4 + A, c + s * 2.75), PT(T, x + 7.4 + A, y), PT(T, x + DX - 7.4, y)])
                    nxt.append(y)
                else:
                    k = int(round((y - YT) / PITCH + 7.5))
                    leaves[k] = (x + 7.4, c + s * 2.75)
        nodes = nxt
    return leaves


leaves = tree(ID, "split")
tree(MX, "combine")
assert sorted(leaves) == list(range(16))

# ---------------------------------------------------------------- arms: leaf -> heater -> leaf
ROLE = {}
for upper in (True, False):
    for u in range(8):
        k = arm_index(u, upper)
        xp, yp = leaves[k]
        y = leaf_y(k)
        xs = X0 + u * DS
        nm = f"A{k:02d}"
        ROLE[k] = nm
        b.wg(f"arm{k}", [DPoint(xp, yp), DPoint(XLT, yp), DPoint(XLT, y), DPoint(xs, y)])
        T = ID if upper else MY
        # heaters in the lower half are mirrored, so their contacts face the bottom pad row
        h = b.place(b.C_HEAT, T, xs, y if upper else 2 * YT - y)
        e1, e2 = b.pinp(h, "elec1"), b.pinp(h, "elec2")
        b.heaters[nm] = sorted([(e1.x, e1.y), (e2.x, e2.y)])
        b.lengths[f"arm{k}"] += HEATER
        b.wg(f"arm{k}", [DPoint(xs + HEATER, y), PT(MX, XLT, y), PT(MX, XLT, yp), PT(MX, xp, yp)])

# ---------------------------------------------------------------- staircase wiring and pads
pads_all, heater_pads = [], []


def half(T, upper, y_pad):
    """Pads and wires of one half, drawn in upper-half coordinates and mapped with T."""
    names = []
    for kk in range(9):
        x = X0 + 5.0 + kk * DS
        b.place(b.C_PAD, T, x, y_pad + 50.0)
        names.append(f"Neuron_{'T' if upper else 'B'}{kk}")
        b.label(f"elec_{USER}_{names[-1]}", T * DPoint(x, y_pad + 50.0))

    def W(pts):
        b.wire([T * DPoint(*p) for p in pts])

    def ends(u):
        k = arm_index(u, upper)
        L, R = sorted([T.inverted() * DPoint(*p) for p in b.heaters[ROLE[k]]], key=lambda q: q.x)
        return L, R

    L0, _ = ends(0)
    W([(L0.x, L0.y), (L0.x, y_pad)])
    for kk in range(1, 9):
        _, R = ends(kk - 1)
        ymid = R.y - 2.0 + PITCH / 2
        xn = X0 + 5.0 + kk * DS
        W([(R.x, R.y), (R.x, ymid), (xn, ymid)])
        if kk < 8:
            L, _ = ends(kk)
            W([(xn, ymid), (L.x, L.y)])
        W([(xn, ymid), (xn, y_pad)])
    for u in range(8):
        k = arm_index(u, upper)
        heater_pads.append(dict(name=ROLE[k], arm=k, between=[names[u], names[u + 1]]))
    pads_all.extend(names)


half(ID, True, Y_PAD_TOP)

# ---------------------------------------------------------------- reference arm, trim heater
XR = X0 - 2 * DS                  # trim heater start
x_up = XC - 35.45 - 25.0
b.wg("ref", [DPoint(67.4, -2.75), DPoint(80.0, -2.75), DPoint(80.0, Y_REF), DPoint(XR, Y_REF)])
hr = b.place(b.C_HEAT, MY, XR, 2 * YT - Y_REF)
e1, e2 = b.pinp(hr, "elec1"), b.pinp(hr, "elec2")
b.heaters["REF"] = sorted([(e1.x, e1.y), (e2.x, e2.y)])
b.lengths["ref"] += HEATER
b.wg("ref", [DPoint(XR + HEATER, Y_REF), DPoint(x_up, Y_REF), DPoint(x_up, YB - 2.35), DPoint(XC - 35.45, YB - 2.35)])

# output coupler, combiner output, output routes
b.place(b.C_BDC, ID, XC, YB)
x_ro = (MX * DPoint(X1 - 7.4, YT)).x
b.wg("sig_out", [DPoint(x_ro, YT), DPoint(x_ro + 20.0, YT), DPoint(x_ro + 20.0, YB + 2.35), DPoint(XC - 35.45, YB + 2.35)])
Y_RB = Y_REF - 40.0
b.wg("out_plus", [DPoint(XC + 35.3, YB + 2.35), DPoint(XC + 55, YB + 2.35), DPoint(XC + 55, Y_RT), DPoint(40, Y_RT), DPoint(40, 127), DPoint(0, 127)])
b.wg("out_minus", [DPoint(XC + 35.3, YB - 2.35), DPoint(XC + 55, YB - 2.35), DPoint(XC + 55, Y_RB), DPoint(40, Y_RB), DPoint(40, -127), DPoint(0, -127)])

# lower half: pad row below the OUT- route
y_pad_low = 2 * YT - (Y_RB - 40.0)            # in mirrored (upper-half) coordinates
half(MY, False, y_pad_low)
# reference trim pads, at the left end of the bottom row
(rl, rr) = sorted([MY.inverted() * DPoint(*p) for p in b.heaters["REF"]], key=lambda q: q.x)
ref_names = []
for kk, x in enumerate((XR + 5.0, XR + 5.0 + DS)):
    b.place(b.C_PAD, MY, x, y_pad_low + 50.0)
    ref_names.append(f"Neuron_R{kk}")
    b.label(f"elec_{USER}_{ref_names[-1]}", MY * DPoint(x, y_pad_low + 50.0))
b.wire([MY * DPoint(rl.x, rl.y), MY * DPoint(rl.x, y_pad_low)])
b.wire([MY * DPoint(rr.x, rr.y), MY * DPoint(rr.x, rr.y + 20), MY * DPoint(XR + 5.0 + DS, rr.y + 20), MY * DPoint(XR + 5.0 + DS, y_pad_low)])
heater_pads.append(dict(name="REF", arm="reference", between=ref_names))
pads_all.extend(ref_names)

opt = f"opt_in_TE_1550_device_{USER}_Neuron16"
b.label(opt, DPoint(0, 0))
b.opt_in.append(opt)
for r, role in ((127.0, "plus"), (0.0, "laser1"), (-127.0, "minus")):
    b.gc_roles[(0, role)] = b.SHIFT * DPoint(0, r)
b.bits.append((0, pya.DBox(-1e4, -1e4, 1e4, 1e4), "", b.SHIFT))
b.blocks.append(dict(bit=0, block="neuron", pads=pads_all, heaters=heater_pads))

r = b.finish(OUT, dft_at=(60.0, 2060.0))
with open(os.path.join(OUT, "perceptron_extracted_netlist.json"), "w") as f:
    json.dump(dict(components=r["ext"]), f, indent=1)
Ls = b.lengths
arm_len = [round(Ls[f"arm{k}"], 3) for k in range(16)]
Y_LEN = 14.8                    # a Y-branch, opt1 to opt2/3 (PDK cell, 15 um)
seg = [Ls[f"split{lv}"] / 2 ** (lv + 1) for lv in range(3)]
signal_path = Ls["sig_in"] + 2 * sum(seg) + arm_len[0] + Ls["sig_out"] + 8 * Y_LEN
mismatch = signal_path - Ls["ref"]
report = dict(top_cell=TOP, die_um=r["die"], siepic_verification_errors=r["n_err"], components=len(r["ext"]),
              waveguide_crossings=r["crossings"], dangling_optical_pins=r["dangling"],
              lengths_um={k: round(v, 3) for k, v in Ls.items()}, arm_lengths_um=arm_len,
              reference_depth_um=Y_REF, signal_path_um=round(signal_path, 3), reference_path_um=round(Ls["ref"], 3),
              path_mismatch_um=round(mismatch, 3), heater_pads=heater_pads, pitch_um=PITCH, staircase_step_um=DS)
with open(os.path.join(OUT, "perceptron_build_report.json"), "w") as f:
    json.dump(report, f, indent=1)
print("die", report["die_um"], "| SiEPIC verification errors:", r["n_err"], "| components", len(r["ext"]),
      "| crossings", len(r["crossings"]), "| dangling", len(r["dangling"]),
      "| arms", min(arm_len), max(arm_len), "| signal", round(signal_path, 3), "ref", round(Ls["ref"], 3), "-> depth for a match:", round(Y_REF - mismatch / 2, 3))
