#!/usr/bin/env python3
"""Layout of the optical full adder chip (paper Fig. 15, hybrid version) for the SiEPIC EBeam
(Applied Nanotools) process with metal heaters.

Floor plan
  * One fibre-array column of grating couplers at x = 0, 127 um pitch, all TE 1550:
        y = +635, +508   loopback (fibre-array alignment)
        y = +381         XOR1  (half adder sum, read by comparator -> X)
        y = +254         carry branch b (input B, AND gate 1)
        y = +127         carry branch a (input A, AND gate 1)
        y =    0         LASER in  (opt_in label)
        y = -127         carry branch c (carry-in, AND gate 2)
        y = -254         carry branch d (X, AND gate 2)
        y = -381         SUM
    The four carry outputs go to photodiodes wired in parallel: their currents add, like the
    paper's single "OR" photoresistor.
  * Two identical half blocks, the second mirrored below the laser row. Each block: two Y-branch
    taps (S2/S3 or S5), two marked AND-branch switches, one Michelson XOR (broadband coupler,
    two arm switches, a bias heater, two Sagnac loop mirrors).
  * Five heaters per block, driven from six pads in a row (top row for block 1, bottom row for
    block 2). Adjacent heaters share a pad, so N heaters need N+1 pads and the metal stays planar;
    each heater sees the voltage difference of its two pads.

Run with the layout environment (gdsfactory + klayout); writes full_adder_chip.gds,
layout_extract.json (waveguide lengths and connectivity for the simulation) and pad_map.json.
"""
import json
import math
import os

import gdsfactory as gf
from gdsfactory.generic_tech import get_generic_pdk

get_generic_pdk().activate()
import cells as C  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
PITCH = 127.0
ROWS = (127.0, 254.0, 381.0)
HEATER = 120.0
USER = "atharvdubey"
BEND90 = gf.components.bend_euler(cross_section=C.XS)
BEND_LEN = float(BEND90.info["length"])


class Log:
    def __init__(self):
        self.len = {}
        self.edges = []
        self.routes = {}

    def route(self, c, p1, p2, key, waypoints=None, extra=0.0):
        kw = dict(cross_section=C.XS)
        if waypoints is not None:
            kw["waypoints"] = waypoints
        r = gf.routing.route_single(c, p1, p2, **kw)
        self.routes.setdefault(c.name, (c, []))[1].append((key, r, p1, p2))
        L = r.length_straights / 1000.0 + r.n_bend90 * BEND_LEN + extra
        self.len[key] = self.len.get(key, 0.0) + L
        self.edges.append((key, round(L, 3)))
        return L


def metal_path(c, pts, width=10.0):
    """Manhattan M2 wire through the given points."""
    for (x0, y0), (x1, y1) in zip(pts[:-1], pts[1:]):
        if abs(x0 - x1) < 1e-9:
            ya, yb = sorted((y0, y1))
            c.add_polygon([(x0 - width / 2, ya - width / 2), (x0 + width / 2, ya - width / 2),
                           (x0 + width / 2, yb + width / 2), (x0 - width / 2, yb + width / 2)], layer=C.M2)
        else:
            xa, xb = sorted((x0, x1))
            c.add_polygon([(xa - width / 2, y0 - width / 2), (xb + width / 2, y0 - width / 2),
                           (xb + width / 2, y0 + width / 2), (xa - width / 2, y0 + width / 2)], layer=C.M2)


def half_block(log, h, names):
    """One half of the adder, drawn for the upper half (y > 0). names: (in1, in2) input names."""
    c = gf.Component(f"half_block_{h}")
    sw = C.mzi_switch(heater_length=HEATER)
    in1, in2 = names
    # ---- grating couplers for this block's three outputs
    gcs = []
    for r in ROWS:
        g = c << C.gc()
        g.dmove((0, r))
        gcs.append(g)
    # ---- AND-branch switches, rotated so the input is on the east and the heater on top
    s_and = []
    for r, g in zip(ROWS[:2], gcs[:2]):
        s = c << sw
        s.drotate(180)
        s.dmove((80.0 - s.ports["o2"].dcenter[0], r - s.ports["o2"].dcenter[1]))
        s_and.append(s)
    log.route(c, gcs[0].ports["opt1"], s_and[0].ports["o2"], f"and_out_{in1}_and")
    log.route(c, gcs[1].ports["opt1"], s_and[1].ports["o2"], f"and_out_{in2}_and")
    x_sw_in = s_and[0].ports["o1"].dcenter[0]
    # ---- taps Sa, Sb on the y = 60 line
    sa = c << C.ybranch()
    sa.dmove((x_sw_in + 70.0, 60.0))
    sb = c << C.ybranch()
    sb.dmove((x_sw_in + 160.0, sa.ports["opt3"].dcenter[1]))      # in line with Sa's lower output: a straight join
    log.route(c, sa.ports["opt2"], s_and[0].ports["o1"], f"and_in_{in1}_and")
    log.route(c, sa.ports["opt3"], sb.ports["opt1"], f"ab{h}")
    log.route(c, sb.ports["opt2"], s_and[1].ports["o1"], f"and_in_{in2}_and")
    # ---- Michelson XOR: mirrored coupler, input left-bottom, output left-top
    xb, yb = sb.ports["opt3"].dcenter[0] + 150.0, 165.0
    bd = c << C.bdc()
    bd.dmirror_y()
    bd.dmove((xb, yb))
    p_in, p_out, p_arm_a, p_arm_b = (bd.ports["opt1"], bd.ports["opt2"], bd.ports["opt3"], bd.ports["opt4"])
    assert p_in.dcenter[1] < p_out.dcenter[1] and p_arm_a.dcenter[1] < p_arm_b.dcenter[1]
    log.route(c, sb.ports["opt3"], p_in, f"sx{h}",
              waypoints=[(p_in.dcenter[0] - 40.0, sb.ports["opt3"].dcenter[1]), (p_in.dcenter[0] - 40.0, p_in.dcenter[1])])
    log.route(c, p_out, gcs[2].ports["opt1"], f"x_out{h}",
              waypoints=[(p_out.dcenter[0] - 12.0, p_out.dcenter[1]), (p_out.dcenter[0] - 12.0, ROWS[2])])
    # arm a (lower): switch below the coupler, then the bias heater, then the loop mirror
    swa = c << sw
    swa.dmirror_y()
    swa.dmove((xb + 120.0 - swa.ports["o1"].dcenter[0], yb - 45.0 - swa.ports["o1"].dcenter[1]))
    log.route(c, p_arm_a, swa.ports["o1"], f"arm_a{h}")
    bias = c << C.heater_wg(length=HEATER)
    bias.connect("o1", swa.ports["o2"])
    log.len[f"arm_a{h}"] += HEATER + 40.0
    # arm b (upper): switch east of the bias heater (heaters must not share an x range, so every
    # heater's wires run straight up to the pads)
    swb = c << sw
    swb.dmirror_y()
    xb_sw = bias.ports["e2"].dcenter[0] + 90.0 + (swb.ports["o1"].dcenter[0] - swb.ports["e1"].dcenter[0])
    swb.dmove((xb_sw - swb.ports["o1"].dcenter[0], yb + 60.0 - swb.ports["o1"].dcenter[1]))
    log.route(c, p_arm_b, swb.ports["o1"], f"arm_b{h}")
    lmb = c << C.loop_mirror()
    lmb.connect("o1", swb.ports["o2"])
    # equalise the arms: put loop a so that arm a's waveguide length equals arm b's
    need = log.len[f"arm_b{h}"] - log.len[f"arm_a{h}"]
    lma = c << C.loop_mirror()
    lma.dmirror_y()
    lma.dmove((bias.ports["o2"].dcenter[0] + max(need, 20.0) - lma.ports["o1"].dcenter[0],
               bias.ports["o2"].dcenter[1] - lma.ports["o1"].dcenter[1]))
    log.route(c, bias.ports["o2"], lma.ports["o1"], f"arm_a{h}")
    # ---- block input port
    c.add_port("in", port=sa.ports["opt1"])
    for k, g in enumerate(gcs):
        c.add_port(f"g{k}", port=g.ports["opt1"])
    # ---- heaters and pads (chain: pad k joins heater k's right contact and heater k+1's left)
    heaters = [(f"{in1}_and", s_and[0]), (f"{in2}_and", s_and[1]), (f"{in1}_x{h}", swa), ("bias", bias), (f"{in2}_x{h}", swb)]
    con = [(nm, sorted([r.ports["e1"].dcenter, r.ports["e2"].dcenter])) for nm, r in heaters]
    for nm, pts in con:
        assert abs(pts[0][1] - pts[1][1]) < 1e-6, nm
    y_pad = ROWS[2] + 170.0
    pads_x = [150.0 + 125.0 * k for k in range(6)]
    # contacts sit on the heater's side facing the pads; wires go up
    (L1, R1), (L2, R2), (L3, R3), (L4, R4), (L5, R5) = [pts for _, pts in con]
    # channel heights: pads fed from the east (2..5) need the westmost pad on the lowest channel
    ych = [0.0, y_pad - 130.0] + [y_pad - 110.0 + 20.0 * (k - 2) for k in range(2, 6)]
    ych[0] = ych[1] + 2.0                                     # pad 0 is reached at ych[0] + 20 (above pad 1's)
    # heater 1 sits under heater 2: its wires escape west around it, then up
    metal_path(c, [L1, (L1[0], ROWS[0] + 58.0), (30.0, ROWS[0] + 58.0), (30.0, ych[0] + 20.0), (pads_x[0], ych[0] + 20.0), (pads_x[0], y_pad)])
    metal_path(c, [R1, (R1[0], ROWS[0] + 78.0), (52.0, ROWS[0] + 78.0), (52.0, ych[1]), (pads_x[1], ych[1]), (pads_x[1], y_pad)])
    metal_path(c, [L2, (L2[0], ych[1])])
    metal_path(c, [R2, (R2[0], ych[2]), (pads_x[2], ych[2]), (pads_x[2], y_pad)])
    metal_path(c, [L3, (L3[0], ych[2]), (pads_x[2], ych[2])])
    for k, (a, b) in ((3, (R3, L4)), (4, (R4, L5))):
        metal_path(c, [a, (a[0], ych[k]), (pads_x[k], ych[k]), (pads_x[k], y_pad)])
        metal_path(c, [b, (b[0], ych[k])])
        metal_path(c, [(a[0], ych[k]), (b[0], ych[k])])
    metal_path(c, [R5, (R5[0], ych[5]), (pads_x[5], ych[5]), (pads_x[5], y_pad)])
    pad_names = [f"P{h}{k}" for k in range(6)]
    for k, x in enumerate(pads_x):
        p = c << C.bond_pad()
        p.dmove((x, y_pad + 50.0))
    c.info["pads"] = [dict(name=pad_names[k], x=pads_x[k], y=y_pad + 50.0) for k in range(6)]
    c.info["heaters"] = [dict(name=nm, between=(pad_names[k], pad_names[k + 1])) for k, (nm, _) in enumerate(heaters)]
    return c


def chip():
    log = Log()
    top = gf.Component("bench_to_pixel_full_adder")
    # laser input and S1
    g_in = top << C.gc()
    s1 = top << C.ybranch()
    s1.dmove((70.0, 0.0))
    log.route(top, g_in.ports["opt1"], s1.ports["opt1"], "gc_s1")
    b1 = top << half_block(log, 1, ("A", "B"))
    log2 = Log()
    b2c = half_block(log2, 2, ("C", "X"))
    b2 = top << b2c
    b2.dmirror_y()
    for k, v in log2.len.items():
        log.len[k] = v
    log.edges += log2.edges
    log.routes.update(log2.routes)
    log.route(top, s1.ports["opt2"], b1.ports["in"], "in1")
    log.route(top, s1.ports["opt3"], b2.ports["in"], "in2")
    # loopback for fibre-array alignment
    lb = []
    for y in (508.0, 635.0):
        g = top << C.gc()
        g.dmove((0, y))
        lb.append(g)
    gf.routing.route_single(top, lb[0].ports["opt1"], lb[1].ports["opt1"], cross_section=C.XS)
    # labels for automated measurement (SiEPIC convention, text layer 10/0)
    top.add_label(f"opt_in_TE_1550_device_{USER}_FullAdder", position=(0, 0), layer=C.TXT)
    top.add_label(f"opt_in_TE_1550_device_{USER}_Loopback", position=(0, 635.0), layer=C.TXT)
    pads = []
    for blk, sign in ((b1.cell, 1), (b2c, -1)):
        for p in blk.info["pads"]:
            pads.append(dict(name=p["name"], x=p["x"], y=sign * p["y"]))
            top.add_label(f"elec_{USER}_FullAdder_{p['name']}", position=(p["x"], sign * p["y"]), layer=C.TXT)
    heaters = list(b1.cell.info["heaters"]) + list(b2c.info["heaters"])
    # floor plan
    bb = top.dbbox()
    m = 30.0
    top.add_polygon([(bb.left - m, bb.bottom - m), (bb.right + m, bb.bottom - m), (bb.right + m, bb.top + m), (bb.left - m, bb.top + m)], layer=C.FP)
    return top, log, pads, heaters


def main(out_dir=HERE):
    top, log, pads, heaters = chip()
    import verify
    probs = []
    for comp, routes in log.routes.values():
        probs += verify.route_overlaps(comp, routes)
    print("waveguide crossing problems:", probs if probs else "none")
    gds = os.path.join(out_dir, "full_adder_chip.gds")
    top.write_gds(gds)
    L = dict(log.len)
    L["mzi_arm"] = float(C.mzi_switch(heater_length=HEATER).info["arm_length"])
    L["loop"] = float(C.loop_mirror().info["loop_length"])
    bb = top.dbbox()
    with open(os.path.join(out_dir, "layout_extract.json"), "w") as f:
        json.dump(dict(lengths_um=L, routes=log.edges, die_um=[bb.width(), bb.height()]), f, indent=1)
    with open(os.path.join(out_dir, "pad_map.json"), "w") as f:
        json.dump(dict(pads=pads, heaters=heaters,
                       drive="Each heater sits between two adjacent pads; set pad potentials so each heater sees its own voltage difference."), f, indent=1)
    print("die", round(bb.width(), 1), "x", round(bb.height(), 1), "um")
    print({k: round(v, 1) for k, v in L.items()})
    return gds


if __name__ == "__main__":
    main()
