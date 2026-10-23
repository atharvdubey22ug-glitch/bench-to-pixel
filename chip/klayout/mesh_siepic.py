#!/usr/bin/env python3
"""
Optical matrix processor: a programmable 8x8 interferometer mesh (Clements layout) on the SiEPIC EBeam
process, the layer of an optical neural network.

    laser -> 1x8 Y-branch tree -> 8 input encoders -> 8 x 8 mesh (28 Mach-Zehnder interferometers) -> outputs
                                   (Y-branch MZIs,     column c couples channels (i, i+1) with
                                    push-pull heaters)  i = c mod 2, 0 <= i <= 6

Inputs. Each channel has a Y-branch Mach-Zehnder with a heater on each arm. Driven push-pull (one arm
up, the other down by the same phase) it sends out cos(delta / 2) of its light with a fixed phase:
a real, signed amplitude, the input value x_j.

Mesh. Eight columns of 2x2 Mach-Zehnder interferometers (two broadband couplers, a heater theta on one
arm) with a heater phi on one input channel: 56 heaters. With every heater set, the mesh multiplies
the eight input fields by an 8x8 unitary matrix. Channels that a column does not couple (0 and 7 in
odd columns) pass straight through, lengthened so that every input-to-output path has the same delay.

Outputs. The SiEPIC Phot1x test rules allow one laser coupler with at most one detector coupler above
it and two below, 127 um apart. Outputs 7, 0 and 1 are routed back to those three couplers; outputs
2 to 6 end in terminators. A layer's eight outputs are read three at a time: the mesh is programmed
so that the wanted rows of the matrix land on outputs 7, 0 and 1 (any unitary can be completed
around three rows, so this costs nothing in accuracy). A process with on-chip photodiodes would put
one on every output and read the layer in one pass.

Heaters and pads. Within a column, the interferometers whose wires go up (upper half) or down (lower
half) sit in two different x slots, so every heater's wires run straight to the pad rows, crossing
waveguides only where they have no heater. The encoders form a staircase for the same reason. All
upper heaters are chained on one row of pads at the top (heater k between pads k and k + 1), all
lower heaters on one row at the bottom. On this process the heaters are TiW thermal phase shifters;
../speed/mesh_speed.py treats the encoders as fast modulators of the same placement.

Outputs of this script: EBeam_heaters_atharvdubey_Mesh8.gds/.oas/.lyrdb, mesh_extracted_netlist.json
(SiEPIC net extraction, with the drawn path of every waveguide for the thermal crosstalk model) and
mesh_build_report.json.
"""
import json
import math
import os
import sys

import pya
from pya import DPoint, DCplxTrans

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from builder import Builder, USER, HEATER  # noqa: E402

OUT = sys.argv[1] if len(sys.argv) > 1 else HERE
TOP = f"EBeam_heaters_{USER}_Mesh8"
N = 8
P = 100.0                         # channel pitch; channel i at y = i P
YL = 3.5 * P                      # laser coupler and tree root
BDC_LEN = 67.5                    # a broadband coupler's group delay as waveguide length (drawn 70.75 um),
                                  # from the PDK model's phase slope; see mesh_sim.path_delays()
# tree (as in the neuron): root Y-branch at X1, levels DX apart, A lead before each turn
X1, DX, A = 100.0, 50.0, 12.0
H = [2 * P, P, P / 2]
XLT = X1 + 2 * DX + 7.4 + A
# encoders: Y-branch MZIs, arms +-AE, heater A on one arm and B on the other, offset in x
XE0, AE, DS = 260.0, 30.0, 340.0
L_EA = 280.0                      # straight arm length
# mesh columns
X_M = XE0 + 4 * DS + 20.0
AM = 20.0                         # MZI arms at +-AM about the coupler
W = 762.0                         # column width
SLOT = 140.0                      # x offset between the two heater slots
Y_RT = (N - 1) * P + 50.0         # top output route (output 7)
Y_RB0, Y_RB1 = -50.0, -70.0       # bottom output routes (outputs 0 and 1)
Y_BUS_T, Y_BUS_B = Y_RT + 40.0, Y_RB1 - 40.0
Y_PAD_T, Y_PAD_B = Y_BUS_T + 80.0, Y_BUS_B - 80.0      # pad centres

b = Builder(TOP)
b.heaters_per_bit = 2 * N + 56
b.bit, b.pre = 0, ""
b.SHIFT = DCplxTrans(1, 0, False, 100.0, -Y_PAD_B + 100.0)
ID = DCplxTrans()
PATHS = []                         # (key, absolute points, length) of every waveguide drawn
HEAT = {}                          # heater name -> dict(x0, y, mirrored, wires: [x_left, x_right], up)


def wg(key, pts):
    L = b.wg(key, [DPoint(*p) for p in pts])
    PATHS.append(dict(key=key, pts=[[round((b.SHIFT * DPoint(*p)).x, 3), round((b.SHIFT * DPoint(*p)).y, 3)] for p in pts], length=L))
    return L


def heater(name, x0, y, up, key):
    """Heater from x0 to x0 + HEATER on the waveguide at y; contacts face up (wires to the top row) or down."""
    h = b.place(b.C_HEAT, ID, x0, y, mirror=not up)
    e1, e2 = b.pinp(h, "elec1"), b.pinp(h, "elec2")
    b.heaters[name] = sorted([(e1.x, e1.y), (e2.x, e2.y)])
    b.lengths[key] = b.lengths.get(key, 0.0) + HEATER
    HEAT[name] = dict(x0=x0, y=y, up=up, contacts=sorted([e1.x, e2.x]), cy=e1.y)
    PATHS.append(dict(key=key, heater=name, pts=[[round((b.SHIFT * DPoint(x0, y)).x, 3), round((b.SHIFT * DPoint(x0, y)).y, 3)],
                                                  [round((b.SHIFT * DPoint(x0 + HEATER, y)).x, 3), round((b.SHIFT * DPoint(x0 + HEATER, y)).y, 3)]],
                      length=HEATER))


# ---------------------------------------------------------------- couplers, tree
GC_ROLE = {YL: "laser", YL + 127.0: "o7", YL - 127.0: "o0", YL - 254.0: "o1"}
for y in GC_ROLE:
    b.place(b.C_GC, ID, 0, y)
wg("laser_in", [(0, YL), (X1 - 7.4, YL)])
nodes = [YL]
for lv in range(3):
    x = X1 + lv * DX
    nxt = []
    for c in nodes:
        b.place(b.C_Y, ID, x, c)
        for s in (1, -1):
            y = c + s * H[lv]
            if lv < 2:
                wg(f"tree{lv}", [(x + 7.4, c + s * 2.75), (x + 7.4 + A, c + s * 2.75), (x + 7.4 + A, y), (x + DX - 7.4, y)])
                nxt.append(y)
            else:
                i = int(round(y / P))
                # leaf: turn onto channel i, run to this channel's encoder
                u = i - 4 if i >= 4 else 3 - i                        # staircase step
                wg(f"ch{i}_in", [(x + 7.4, c + s * 2.75), (XLT, c + s * 2.75), (XLT, y), (XE0 + u * DS, y)])
    nodes = nxt

# ---------------------------------------------------------------- encoders
ENC = {}
for i in range(N):
    up = i >= 4
    u = i - 4 if up else 3 - i
    y = i * P
    xin = XE0 + u * DS
    xe = xin + 7.4
    b.place(b.C_Y, ID, xe, y)
    xa0 = xe + 7.4 + 10.0
    xa1 = xa0 + L_EA
    xe2 = xa1 + 17.4
    b.place(b.C_Y, ID, xe2, y, rot=180)
    s = 1 if up else -1              # first heater on the arm on the wiring side
    for arm, (sgn, slot) in {"A": (s, 0), "B": (-s, 1)}.items():
        ya = y + sgn * AE
        key = f"enc{i}_{arm}"
        x_h = xa0 + 10.0 + slot * SLOT
        wg(key, [(xe + 7.4, y + sgn * 2.75), (xa0, y + sgn * 2.75), (xa0, ya), (x_h, ya)])
        heater(f"E{i}{arm}", x_h, ya, up, key)
        wg(key, [(x_h + HEATER, ya), (xa1, ya), (xa1, y + sgn * 2.75), (xe2 - 7.4, y + sgn * 2.75)])
    wg(f"ch{i}_enc_out", [(xe2 + 7.4, y), (X_M, y)])
    ENC[i] = dict(step=u, x_in=xin, x_out=xe2 + 7.4)

# ---------------------------------------------------------------- mesh
# (lower channel i, wires up?, slot) per column
COLS = {0: [(0, False, 1), (2, False, 0), (4, True, 0), (6, True, 1)],
        1: [(1, False, 1), (3, True, 0), (5, True, 1)]}
MZIS = []


def mzi(c, i, up, slot):
    Xc = X_M + c * W
    yc = (i + 0.5) * P
    lo, hi = i * P, (i + 1) * P
    # phi heater on the channel on the wiring side: the upper channel if wires go up, else the lower
    ych = hi if up else lo
    other = lo if up else hi
    xph = Xc + 10.0 + slot * SLOT
    kph, kot = f"c{c}_ch{(i + 1) if up else i}", f"c{c}_ch{i if up else i + 1}"
    wg(kph, [(Xc, ych), (xph, ych)])
    heater(f"F{c}_{i}", xph, ych, up, kph)
    xf = Xc + 280.0
    xv = xf + 10.0
    xb1 = xv + 10.0 + 35.45
    sgn_ph = 1 if up else -1
    wg(kph, [(xph + HEATER, ych), (xv, ych), (xv, yc + sgn_ph * 2.35), (xb1 - 35.45, yc + sgn_ph * 2.35)])
    wg(kot, [(Xc, other), (xv, other), (xv, yc - sgn_ph * 2.35), (xb1 - 35.45, yc - sgn_ph * 2.35)])
    b.place(b.C_BDC, ID, xb1, yc)
    xb2 = xb1 + 370.75
    b.place(b.C_BDC, ID, xb2, yc)
    xh = xb1 + 45.3 + 10.0 + slot * SLOT
    ka, kb = f"c{c}_m{i}_theta_arm", f"c{c}_m{i}_plain_arm"
    for sgn, key in ((sgn_ph, ka), (-sgn_ph, kb)):
        ya = yc + sgn * AM
        p0 = [(xb1 + 35.3, yc + sgn * 2.35), (xb1 + 45.3, yc + sgn * 2.35), (xb1 + 45.3, ya)]
        p1 = [(xb2 - 45.45, ya), (xb2 - 45.45, yc + sgn * 2.35), (xb2 - 35.45, yc + sgn * 2.35)]
        if key == ka:
            wg(key, p0 + [(xh, ya)])
            heater(f"T{c}_{i}", xh, ya, up, key)
            wg(key, [(xh + HEATER, ya)] + p1)
        else:
            wg(key, p0 + p1)
    xe = Xc + W
    for ch, sgn in ((i, -1), (i + 1, 1)):
        wg(f"c{c}_out{ch}", [(xb2 + 35.3, yc + sgn * 2.35), (xb2 + 45.3, yc + sgn * 2.35), (xb2 + 45.3, ch * P), (xe, ch * P)])
    MZIS.append(dict(col=c, i=i, up=up, slot=slot, theta=f"T{c}_{i}", phi=f"F{c}_{i}", phi_channel=(i + 1) if up else i))


for c in range(N):
    for (i, up, slot) in COLS[c % 2]:
        mzi(c, i, up, slot)

# drawn length of one channel through an interferometer column (both channels are equal)
L_MZI = b.lengths["c0_ch1"] + b.lengths["c0_m0_theta_arm"] + b.lengths["c0_out1"] + 2 * BDC_LEN


def passthrough(c, ch, h):
    """Straight channel through odd column c with a bump of height h (toward the mesh) for equal delay."""
    Xc = X_M + c * W
    y = ch * P
    sg = 1 if ch == 0 else -1
    xa = Xc + 380.0
    xb = xa + 60.0
    return wg(f"c{c}_pass{ch}", [(Xc, y), (xa, y), (xa, y + sg * h), (xb, y + sg * h), (xb, y), (Xc + W, y)])


# bump height: L(h) is linear in h, two trial lengths fix it
def trial_len(h):
    c = b.ly.create_cell("Waveguide", "EBeam", {"path": pya.DPath([DPoint(0, 0), DPoint(380, 0), DPoint(380, h), DPoint(440, h), DPoint(440, 0), DPoint(W, 0)], 0.5),
                                                "waveguide_type": "Strip TE 1550 nm, w=500 nm"})
    L = float(c.display_title().split("_")[-1])
    b.ly.delete_cell(c.cell_index())
    return L


l1, l2 = trial_len(30.0), trial_len(60.0)
H_PASS = 30.0 + (L_MZI - l1) * 30.0 / (l2 - l1)
for c in range(1, N, 2):
    for ch in (0, N - 1):
        passthrough(c, ch, H_PASS)

# ---------------------------------------------------------------- outputs
X_END = X_M + N * W
for ch in range(2, N - 1):
    b.place(b.C_TERM, ID, X_END + 10.0, ch * P, rot=180)
    wg(f"out{ch}_term", [(X_END, ch * P), (X_END + 10.0, ch * P)])
x7 = X_END + 40.0
wg("route_o7", [(X_END, 7 * P), (x7, 7 * P), (x7, Y_RT), (40.0, Y_RT), (40.0, YL + 127.0), (0, YL + 127.0)])
xa, xb = X_END + 40.0, X_END + 70.0
wg("route_o0", [(X_END, 0.0), (xa, 0.0), (xa, Y_RB0), (55.0, Y_RB0), (55.0, YL - 127.0), (0, YL - 127.0)])
wg("route_o1", [(X_END, P), (xb, P), (xb, Y_RB1), (25.0, Y_RB1), (25.0, YL - 254.0), (0, YL - 254.0)])

# ---------------------------------------------------------------- wiring: one chain per side
pads_all, heater_pads = [], []


def chain(up):
    names = [n for n, h in HEAT.items() if h["up"] == up]
    names.sort(key=lambda n: HEAT[n]["contacts"][0])
    y_bus = Y_BUS_T if up else Y_BUS_B
    y_pad = Y_PAD_T if up else Y_PAD_B
    side = "T" if up else "B"
    wires = []
    for n in names:
        h = HEAT[n]
        for xc in h["contacts"]:
            b.wire([DPoint(xc, h["cy"]), DPoint(xc, y_bus)])
        wires.append(h["contacts"])
    # pad k: left contact of heater k joined with the right contact of heater k - 1
    pad_x = [wires[0][0]] + [0.5 * (wires[k - 1][1] + wires[k][0]) for k in range(1, len(wires))] + [wires[-1][1]]
    for k, x in enumerate(pad_x):
        if 0 < k < len(wires):
            b.wire([DPoint(wires[k - 1][1], y_bus), DPoint(wires[k][0], y_bus)])
        b.wire([DPoint(x, y_bus), DPoint(x, y_pad)])
        b.place(b.C_PAD, ID, x, y_pad)
        nm = f"Mesh_{side}{k:02d}"
        b.label(f"elec_{USER}_{nm}", DPoint(x, y_pad))
        pads_all.append(nm)
    for k, n in enumerate(names):
        heater_pads.append(dict(name=n, between=[f"Mesh_{side}{k:02d}", f"Mesh_{side}{k + 1:02d}"]))
    return pad_x


pad_x_top = chain(True)
pad_x_bot = chain(False)

opt = f"opt_in_TE_1550_device_{USER}_Mesh8"
b.label(opt, DPoint(0, YL))
b.opt_in.append(opt)
for y, role in GC_ROLE.items():
    b.gc_roles[(0, role)] = b.SHIFT * DPoint(0, y)
b.bits.append((0, pya.DBox(-1e5, -1e5, 1e5, 1e5), "", b.SHIFT))
b.blocks.append(dict(bit=0, block="mesh", pads=pads_all, heaters=heater_pads))

r = b.finish(OUT, dft_at=(60.0, (b.SHIFT * DPoint(0, Y_PAD_T)).y))

# attach drawn paths to the extracted waveguides (matched by bounding-box centre and length)
ext = r["ext"]
used = set()
for e in ext:
    if e["kind"] not in ("Waveguide", "wg_heater"):
        continue
    best, bd = None, 1e9
    for k, p in enumerate(PATHS):
        if k in used:
            continue
        if (e["kind"] == "wg_heater") != ("heater" in p):
            continue
        if e["kind"] == "wg_heater" and p["heater"] != e.get("role"):
            continue
        xs, ys = [q[0] for q in p["pts"]], [q[1] for q in p["pts"]]
        cx, cy = 0.5 * (min(xs) + max(xs)), 0.5 * (min(ys) + max(ys))
        d = abs(cx - e["center"][0]) + abs(cy - e["center"][1]) + (abs(p["length"] - e.get("length_um", p["length"])) if e["kind"] == "Waveguide" else 0)
        if d < bd:
            best, bd = k, d
    assert bd < 2.0, (e, bd)
    used.add(best)
    e["path"] = PATHS[best]["pts"]
    e["key"] = PATHS[best]["key"]
assert len(used) == len(PATHS), (len(used), len(PATHS))
with open(os.path.join(OUT, "mesh_extracted_netlist.json"), "w") as f:
    json.dump(dict(components=ext, mzis=MZIS, channel_pitch_um=P), f, indent=1)

Ls = b.lengths
pass_len = Ls["c1_pass0"] / 1.0
report = dict(top_cell=TOP, die_um=r["die"], siepic_verification_errors=r["n_err"], components=len(ext),
              waveguide_crossings=r["crossings"], dangling_optical_pins=r["dangling"],
              heaters=len(HEAT), pads=len(pads_all), pads_top=len(pad_x_top), pads_bottom=len(pad_x_bot),
              min_pad_pitch_um=round(min(min(b_ - a_ for a_, b_ in zip(px[:-1], px[1:])) for px in (pad_x_top, pad_x_bot)), 2),
              channel_pitch_um=P, column_width_um=W, mzi_channel_path_um=round(L_MZI, 3), passthrough_bump_um=round(H_PASS, 3),
              passthrough_path_um=round(pass_len, 3), coupler_delay_as_length_um=BDC_LEN,
              encoder_arm_lengths_um={f"E{i}": [round(Ls[f"enc{i}_A"], 3), round(Ls[f"enc{i}_B"], 3)] for i in range(N)},
              lengths_um={k: round(v, 3) for k, v in Ls.items()}, heater_pads=heater_pads, mzis=MZIS)
with open(os.path.join(OUT, "mesh_build_report.json"), "w") as f:
    json.dump(report, f, indent=1)
print("die", report["die_um"], "| SiEPIC verification errors:", r["n_err"], "| components", len(ext),
      "| crossings", len(r["crossings"]), "| dangling", len(r["dangling"]), "| heaters", len(HEAT),
      "| pads", len(pads_all), "min pitch", report["min_pad_pitch_um"], "| MZI path", round(L_MZI, 3), "pass", round(pass_len, 3), "bump", round(H_PASS, 2))
