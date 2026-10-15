#!/usr/bin/env python3
"""
Optical full adder (paper Fig. 15, hybrid) as a native KLayout / SiEPIC-Tools layout for the
SiEPIC EBeam process (Applied Nanotools), built from the EBeam library:

  GC_TE_1550_8degOxide_BB   grating couplers (fibre array, 127 um pitch)
  ebeam_y_1550              Y-branches (S1, taps, loop mirrors)
  ebeam_bdc_te1550          broadband directional couplers (Michelson XORs, Mach-Zehnder switches)
  ebeam_terminator_te1550   terminators on unused coupler ports
  wg_heater (PCell)         TiW heaters (switch arms, Michelson bias)
  Waveguide (PCell)         every waveguide, 'Strip TE 1550 nm, w=500 nm'
  ebeam_BondPad (EBeam_Beta) probe pads

Same floor plan as ../layout.py (gdsfactory). Outputs, in this folder:
  EBeam_heaters_atharvdubey_FullAdder.gds / .oas   layout (the .oas is the fabrication export)
  EBeam_heaters_atharvdubey_FullAdder.lyrdb        SiEPIC functional verification report
  netlist.sp                                        SiEPIC SPICE netlist of the drawn circuit
  build_report.json                                 waveguide lengths, heater/pad map, check counts

Run:  pip install klayout siepic siepic_ebeam_pdk ; python3 full_adder_siepic.py
Open the .gds in KLayout with SiEPIC-Tools and the EBeam technology to inspect or edit it.
"""
import json
import math
import os

import pya
from pya import DPoint, DPath, DCplxTrans, DTrans

import SiEPIC  # noqa: F401
import siepic_ebeam_pdk  # noqa: F401  (registers the EBeam technology and libraries)
from SiEPIC.utils.layout import new_layout
from SiEPIC.extend import to_itype  # noqa: F401

HERE = os.path.dirname(os.path.abspath(__file__))
USER = "atharvdubey"
TOP = f"EBeam_heaters_{USER}_FullAdder"
WG = "Strip TE 1550 nm, w=500 nm"
ROWS = (127.0, 254.0, 381.0)
HEATER = 120.0
SHIFT = DCplxTrans(1, 0, False, 120.0, 860.0)      # design origin -> positive floor-plan coordinates

cell, ly = new_layout("EBeam", TOP, GUI=False, overwrite=True)
dbu = ly.dbu
L_SI = ly.layer(ly.TECHNOLOGY["Si"])
L_M2 = ly.layer(ly.TECHNOLOGY["M2_router"])
L_TXT = ly.layer(ly.TECHNOLOGY["Text"])

C_GC = ly.create_cell2("GC_TE_1550_8degOxide_BB", "EBeam")
C_Y = ly.create_cell2("ebeam_y_1550", "EBeam")
C_BDC = ly.create_cell2("ebeam_bdc_te1550", "EBeam")
C_TERM = ly.create_cell2("ebeam_terminator_te1550", "EBeam")
C_PAD = ly.create_cell2("ebeam_BondPad", "EBeam_Beta")
C_HEAT = ly.create_cell("wg_heater", "EBeam", {"length": HEATER, "mh_width": 4, "waveguide_type": WG})

lengths = {}
heaters = {}


def place(c, T, x=0.0, y=0.0, rot=0, mirror=False):
    t = SHIFT * T * DCplxTrans(1, rot, mirror, x, y)
    return cell.insert(pya.DCellInstArray(c.cell_index(), t))


def pinp(inst, name):
    """Pin centre in design coordinates (micron, before SHIFT)."""
    p = inst.find_pin(name)
    return SHIFT.inverted() * DPoint(p.center.x * dbu, p.center.y * dbu)


def wg(key, pts_design):
    """Waveguide PCell along a Manhattan path given in design coordinates; returns its length."""
    pts = [SHIFT * p for p in pts_design]
    c = ly.create_cell("Waveguide", "EBeam", {"path": DPath(pts, 0.5), "waveguide_type": WG})
    cell.insert(pya.CellInstArray(c.cell_index(), pya.Trans()))
    L = float(c.display_title().split("_")[-1])
    lengths[key] = lengths.get(key, 0.0) + L
    return L


def wire(pts, width=10.0):
    """Manhattan M2 wire in design coordinates."""
    for a, b in zip(pts[:-1], pts[1:]):
        a, b = SHIFT * a, SHIFT * b
        if abs(a.x - b.x) < 1e-9:
            box = pya.DBox(a.x - width / 2, min(a.y, b.y) - width / 2, a.x + width / 2, max(a.y, b.y) + width / 2)
        else:
            box = pya.DBox(min(a.x, b.x) - width / 2, a.y - width / 2, max(a.x, b.x) + width / 2, a.y + width / 2)
        cell.shapes(L_M2).insert(box)


def label(text, p, size=5.0):
    q = SHIFT * p
    t = pya.Text(text, pya.Trans(pya.Trans.R0, int(round(q.x / dbu)), int(round(q.y / dbu))))
    cell.shapes(L_TXT).insert(t).text_size = size / dbu


def P(T, x, y):
    return T * DPoint(x, y)


# ---------------------------------------------------------------- Mach-Zehnder switch
# local frame: input bdc1.opt1 at (-35.45, 2.35) facing west, output bdc2.opt3 facing east;
# heater on the lower arm (y = -20); arms are mirror images, so equal in length.
D = 240.6


def switch(name, T):
    b1 = place(C_BDC, T, 0, 0)
    b2 = place(C_BDC, T, D, 0)
    h = place(C_HEAT, T, 60.0, -20.0)
    t1 = place(C_TERM, T, -35.45, -2.35)
    t2 = place(C_TERM, T, D + 35.3, -2.35, rot=180)
    wg(f"{name}_arm_top", [P(T, 35.3, 2.35), P(T, 45.3, 2.35), P(T, 45.3, 20), P(T, D - 45.45, 20), P(T, D - 45.45, 2.35), P(T, D - 35.45, 2.35)])
    wg(f"{name}_arm_heater", [P(T, 35.3, -2.35), P(T, 45.3, -2.35), P(T, 45.3, -20), P(T, 60, -20)])
    wg(f"{name}_arm_heater", [P(T, 60 + HEATER, -20), P(T, D - 45.45, -20), P(T, D - 45.45, -2.35), P(T, D - 35.45, -2.35)])
    lengths[f"{name}_arm_heater"] += HEATER
    e1, e2 = pinp(h, "elec1"), pinp(h, "elec2")
    heaters[name] = sorted([(e1.x, e1.y), (e2.x, e2.y)])
    return dict(inp=P(T, -35.45, 2.35), out=P(T, D + 35.3, 2.35), insts=(b1, b2, h, t1, t2))


def loop_mirror(name, T):
    """Y-branch at the local origin (input opt1 at (-7.4, 0) facing west) with a rectangular loop."""
    y = place(C_Y, T, 0, 0)
    pts = [(7.4, 2.75), (20, 2.75), (20, 15), (45, 15), (45, -15), (20, -15), (20, -2.75), (7.4, -2.75)]
    wg(f"{name}_loop", [P(T, *p) for p in pts])
    return P(T, -7.4, 0)


# ---------------------------------------------------------------- half block
# Each block is one automated-test circuit (SiEPIC Phot1x design-for-test rules: one laser
# coupler with at most one detector coupler above it and two below, 127 um pitch):
#     y = 381  XOR output            (detector above the laser)
#     y = 254  LASER in              (opt_in label)
#     y = 127  AND branch n1         (detector)
#     y =   0  AND branch n2         (detector)
# Block 1 (inputs A, B) is the paper's half adder; block 2 (inputs Cin, X) is the second half
# adder of the textbook full adder (Fig. 12); the carries are added on the photodiodes.
def block(h, T, n1, n2):
    gcs = [place(C_GC, T, 0, r) for r in (0.0, 127.0, 254.0, 381.0)]
    # AND switches rotated 180: output (west) at x = 80 on their coupler rows, heater on top
    sw = {}
    for r, n in ((127.0, n1), (0.0, n2)):
        s = switch(f"{n}_and", T * DCplxTrans(1, 180, False, 80 + D + 35.3, r + 2.35))
        sw[n] = s
        wg(f"and_out_{n}_and", [P(T, 0, r), P(T, 80, r)])
    x_in = (T.inverted() * sw[n1]["inp"]).x
    A, B = x_in + 70.0, x_in + 160.0
    ya, yb_ = 194.0, 196.75                      # Sa centre; Sb centre in line with Sa's upper output
    place(C_Y, T, A, ya)
    place(C_Y, T, B, yb_)
    wg(f"in{h}", [P(T, 0, 254), P(T, A - 37.4, 254), P(T, A - 37.4, ya), P(T, A - 7.4, ya)])
    wg(f"and_in_{n1}_and", [P(T, A + 7.4, ya - 2.75), P(T, A + 17.4, ya - 2.75), P(T, A + 17.4, 127.0), P(T, x_in, 127.0)])
    wg(f"ab{h}", [P(T, A + 7.4, ya + 2.75), P(T, B - 7.4, yb_)])
    wg(f"and_in_{n2}_and", [P(T, B + 7.4, yb_ - 2.75), P(T, B + 17.4, yb_ - 2.75), P(T, B + 17.4, 0.0), P(T, x_in, 0.0)])
    # Michelson: coupler mirrored so opt1 (input) is lower-left, opt2 (output) upper-left
    xb, yb = B + 7.4 + 150.0, 262.0
    place(C_BDC, T, xb, yb, mirror=True)
    p_in, p_out = (xb - 35.45, yb - 2.35), (xb - 35.45, yb + 2.35)
    p_a, p_b = (xb + 35.3, yb - 2.35), (xb + 35.3, yb + 2.35)
    wg(f"sx{h}", [P(T, B + 7.4, yb_ + 2.75), P(T, p_in[0] - 40, yb_ + 2.75), P(T, p_in[0] - 40, p_in[1]), P(T, *p_in)])
    wg(f"x_out{h}", [P(T, *p_out), P(T, p_out[0] - 12, p_out[1]), P(T, p_out[0] - 12, 381.0), P(T, 0, 381.0)])
    # arm a: switch (mirrored: heater on top), bias heater, loop mirror
    xa_in, ya2 = xb + 120.0, yb - 45.0
    swa = switch(f"{n1}_x{h}", T * DCplxTrans(1, 0, True, xa_in + 35.45, ya2 + 2.35))
    wg(f"arm_a{h}", [P(T, *p_a), P(T, p_a[0] + 42.35, p_a[1]), P(T, p_a[0] + 42.35, ya2), P(T, xa_in, ya2)])
    xo = (T.inverted() * swa["out"]).x
    hb = place(C_HEAT, T, xo, ya2)
    e1, e2 = pinp(hb, "elec1"), pinp(hb, "elec2")
    heaters[f"bias{h}"] = sorted([(e1.x, e1.y), (e2.x, e2.y)])
    lengths[f"arm_a{h}"] += HEATER
    # arm b: switch east of the bias heater, then loop mirror
    xb_in, yb2 = xo + HEATER + 70.0, yb + 60.0
    swb = switch(f"{n2}_x{h}", T * DCplxTrans(1, 0, True, xb_in + 35.45, yb2 + 2.35))
    wg(f"arm_b{h}", [P(T, *p_b), P(T, p_b[0] + 26, p_b[1]), P(T, p_b[0] + 26, yb2), P(T, xb_in, yb2)])
    xbo = (T.inverted() * swb["out"]).x
    loop_mirror(f"lmb{h}", T * DCplxTrans(1, 0, False, xbo + 7.4, yb2))
    extra = max(lengths[f"arm_b{h}"] - lengths[f"arm_a{h}"], 20.0)
    wg(f"arm_a{h}", [P(T, xo + HEATER, ya2), P(T, xo + HEATER + extra, ya2)])
    loop_mirror(f"lma{h}", T * DCplxTrans(1, 0, False, xo + HEATER + extra + 7.4, ya2))
    # labels
    label(f"opt_in_TE_1550_device_{USER}_FullAdderHalf{h}", T * DPoint(0, 254.0))
    roles = {381.0: "xor" if h == 1 else "sum", 254.0: f"laser{h}", 127.0: f"c_{'a' if h == 1 else 'c'}", 0.0: f"c_{'b' if h == 1 else 'd'}"}
    for r, nm in roles.items():
        gc_roles[nm] = T * DPoint(0, r)
    # ---- pads and wires (chain: pad k joins heater k's right contact and heater k+1's left)
    y_pad = 381.0 + 170.0
    pads_x = [150.0 + 125.0 * k for k in range(6)]
    for k, x in enumerate(pads_x):
        place(C_PAD, T, x, y_pad + 50.0)
        label(f"elec_{USER}_FullAdder_P{h}{k}", T * DPoint(x, y_pad + 50.0))
    order = [f"{n2}_and", f"{n1}_and", f"{n1}_x{h}", f"bias{h}", f"{n2}_x{h}"]
    loc = [[T.inverted() * DPoint(*p) for p in heaters[nm]] for nm in order]
    (L1, R1), (L2, R2), (L3, R3), (L4, R4), (L5, R5) = [sorted(v, key=lambda q: q.x) for v in loc]
    ych = [0.0, y_pad - 130.0] + [y_pad - 110.0 + 20.0 * (k - 2) for k in range(2, 6)]
    ych[0] = ych[1] + 2.0
    ych[2] = y_pad - 150.0          # below pad 1's channel, so heater 2's right wire clears pad 1's wire

    def W(pts):
        wire([T * DPoint(*p) if isinstance(p, tuple) else T * p for p in pts])

    # heater 1 (row 0) sits under heater 2 (row 127): its wires escape west, then up
    W([L1, (L1.x, 58.0), (30.0, 58.0), (30.0, ych[0] + 20.0), (pads_x[0], ych[0] + 20.0), (pads_x[0], y_pad)])
    W([R1, (R1.x, 78.0), (52.0, 78.0), (52.0, ych[1]), (pads_x[1], ych[1]), (pads_x[1], y_pad)])
    W([L2, (L2.x, ych[1])])
    W([R2, (R2.x, ych[2]), (pads_x[2], ych[2]), (pads_x[2], y_pad)])
    W([L3, (L3.x, ych[2]), (pads_x[2], ych[2])])
    for k, (a, b) in ((3, (R3, L4)), (4, (R4, L5))):
        W([a, (a.x, ych[k]), (pads_x[k], ych[k]), (pads_x[k], y_pad)])
        W([b, (b.x, ych[k])])
        W([(a.x, ych[k]), (b.x, ych[k])])
    W([R5, (R5.x, ych[5]), (pads_x[5], ych[5]), (pads_x[5], y_pad)])
    return dict(pads=[f"P{h}{k}" for k in range(6)],
                heaters=[dict(name=nm, between=[f"P{h}{k}", f"P{h}{k + 1}"]) for k, nm in enumerate(order)])


gc_roles = {}
ID = DCplxTrans()
b1 = block(1, ID, "A", "B")
b2 = block(2, DCplxTrans(1, 0, False, 0, -762.0), "C", "X")
# loopback for fibre-array alignment
place(C_GC, ID, 0, 508.0)
place(C_GC, ID, 0, 635.0)
wg("loopback", [DPoint(0, 508), DPoint(20, 508), DPoint(20, 635), DPoint(0, 635)])
label(f"opt_in_TE_1550_device_{USER}_Loopback", DPoint(0, 635))
label("DFT=DFT_Phot1x", DPoint(-40, 700), 10)

# floor plan around everything
bb = cell.dbbox()
fp = ly.create_cell("FloorPlan")
fp.shapes(ly.layer(ly.TECHNOLOGY["FloorPlan"])).insert(pya.DBox(0, 0, math.ceil(bb.right + 40), math.ceil(bb.top + 40)))
cell.insert(pya.CellInstArray(fp.cell_index(), pya.Trans()))
assert bb.left > 40 and bb.bottom > 40, bb

# ---------------------------------------------------------------- save, verify, netlist
gds = os.path.join(HERE, TOP + ".gds")
ly.write(gds)
from SiEPIC.scripts import export_layout  # noqa: E402
export_layout(cell, HERE, TOP, format="oas", screenshot=False)
from SiEPIC.verification import layout_check  # noqa: E402
rdb = os.path.join(HERE, TOP + ".lyrdb")
n_err = layout_check(cell=cell, verbose=False, GUI=False, file_rdb=rdb)
for h in (1, 2):
    sp = cell.spice_netlist_export(verbose=False, opt_in_selection_text=[f"opt_in_TE_1550_device_{USER}_FullAdderHalf{h}"])
    with open(os.path.join(HERE, f"netlist_half{h}.sp"), "w") as f:
        f.write(sp[0] if isinstance(sp, tuple) else str(sp))

# ---------------------------------------------------------------- extracted netlist for SAX
# Connectivity comes from SiEPIC's own net extraction on the drawn layout (pins that touch).
nets, comps = cell.identify_nets()
inv = SHIFT.inverted()


def dpt(p):
    return inv * DPoint(p.x * dbu, p.y * dbu)


heater_role = {}
for nm, pts in heaters.items():
    heater_role[nm] = DPoint(0.5 * (pts[0][0] + pts[1][0]), pts[0][1])
ext = []
for c in comps:
    name = c.basic_name or c.component
    pins = {p.pin_name: p.net.idx for p in c.pins if p.type == 1 or str(p.pin_name).startswith("opt")}
    item = dict(idx=c.idx, kind=name, pins=pins, center=[round(dpt(c.center).x, 3), round(dpt(c.center).y, 3)])
    prm = dict(kv.split("=", 1) for kv in (c.params or "").split() if "=" in kv)
    if "wg_length" in prm:
        v = prm["wg_length"].rstrip("u")
        item["length_um"] = float(v) if prm["wg_length"].endswith("u") else float(v) * 1e6
    if name == "wg_heater":
        cc = dpt(c.center)
        item["role"] = min(heater_role, key=lambda k: (heater_role[k].x - cc.x) ** 2 + (heater_role[k].y - cc.y) ** 2)
    if name.startswith("GC_"):
        cc = dpt(c.center)
        for role, p in gc_roles.items():
            if abs(p.y - cc.y) < 20 and abs(p.x - cc.x) < 60:
                item["role"] = role
        if "role" not in item:
            item["role"] = f"gc_{int(round(cc.y))}"
    ext.append(item)
roles = [e.get("role") for e in ext if e["kind"] == "wg_heater"]
assert len(set(roles)) == len(roles) == 10, roles

# ---------------------------------------------------------------- independent checks
# 1) waveguide crossings: silicon shared between two different instances must sit at a pin
li = ly.layer(1, 0)
insts = [i for i in cell.each_inst() if i.cell.name != "FloorPlan"]
regs = [pya.Region(i.cell.begin_shapes_rec(li)).transformed(i.cplx_trans) for i in insts]
all_pins = [p.center for c in comps for p in c.pins]
crossings = []
for i in range(len(insts)):
    for j in range(i + 1, len(insts)):
        if not regs[i].bbox().overlaps(regs[j].bbox()):
            continue
        ov = (regs[i] & regs[j]).merged()
        for poly in ov.each():
            cc = poly.bbox().center()
            if min((cc - p).length() for p in all_pins) > 2000:
                q = dpt(cc)
                crossings.append([round(q.x, 1), round(q.y, 1)])
# 2) every optical pin connected except the grating couplers' fibre side
dangling = [(e["kind"], e["center"]) for e in ext for pn, nid in e["pins"].items()
            if sum(1 for f in ext for q in f["pins"].values() if q == nid) < 2]

with open(os.path.join(HERE, "extracted_netlist.json"), "w") as f:
    json.dump(dict(components=ext, gc_roles={k: [v.x, v.y] for k, v in gc_roles.items()}), f, indent=1)
fpb = fp.dbbox()
report = dict(top_cell=TOP, die_um=[fpb.width(), fpb.height()], siepic_verification_errors=n_err,
              components=len(ext), waveguide_crossings=crossings, dangling_optical_pins=dangling,
              lengths_um={k: round(v, 3) for k, v in lengths.items()},
              pads={"block1": b1["pads"], "block2": b2["pads"]}, heater_pads=b1["heaters"] + b2["heaters"])
with open(os.path.join(HERE, "build_report.json"), "w") as f:
    json.dump(report, f, indent=1)
print("die", report["die_um"], "| SiEPIC verification errors:", n_err, "| components", len(ext),
      "| crossings", len(crossings), "| dangling optical pins", len(dangling))
