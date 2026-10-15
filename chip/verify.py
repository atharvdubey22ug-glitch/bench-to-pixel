#!/usr/bin/env python3
"""Checks on full_adder_chip.gds.

1. Design rules of the SiEPIC EBeam DRC deck (siepic_ebeam_pdk drc/SiEPIC_EBeam_DRC.lydrc):
   Si width >= 60 nm, Si space >= 70 nm, M1 width/space >= 3 um, M2 width >= 5 um, M2 space >= 8 um,
   M_Open width/space >= 10 um, M2-over-M1 overlap >= 3 um, every Si shape inside the FloorPlan.
2. Electrical: each merged M2 net carries exactly one pad opening; each heater (M1) touches exactly
   two different nets; no metal within 10 um of a grating coupler.
3. Optical: every routed waveguide touches other silicon only at its two end ports (no crossings).
"""
import json
import os
import sys

import klayout.db as kdb

HERE = os.path.dirname(os.path.abspath(__file__))


def region(ly, cell, layer):
    return kdb.Region(cell.begin_shapes_rec(ly.layer(*layer)))


def drc(gds):
    ly = kdb.Layout()
    ly.read(gds)
    top = ly.top_cell()
    um = lambda v: int(round(v / ly.dbu))  # noqa: E731
    si, m1, m2, mo, fp = (region(ly, top, l).merged() for l in ((1, 0), (11, 0), (12, 0), (13, 0), (99, 0)))
    tol = 0.002
    res = {}
    # angle limits follow the deck (80 deg for Si, 70 deg for metal): curved waveguide edges are not flagged
    res["Si_width"] = si.width_check(um(0.06 - tol), False, kdb.Metrics.Euclidian, 80).count()
    res["Si_space"] = si.space_check(um(0.07 - tol), False, kdb.Metrics.Euclidian, 80).count()
    res["M1_width"] = m1.width_check(um(3.0 - tol), False, kdb.Metrics.Euclidian, 70).count()
    res["M1_space"] = m1.space_check(um(3.0 - tol)).count()
    res["M2_width"] = m2.width_check(um(5.0 - tol), False, kdb.Metrics.Euclidian, 70).count()
    res["M2_space"] = m2.space_check(um(8.0 - tol)).count()
    res["MOpen_width"] = mo.width_check(um(10.0 - tol)).count()
    res["MOpen_space"] = mo.space_check(um(10.0 - tol)).count()
    res["M2_M1_overlap"] = m2.overlap_check(m1, um(3.0 - tol)).count()
    res["Si_outside_floorplan"] = (si - fp).count()
    # electrical nets
    nets = list(m2.each())
    pads_per_net = [(kdb.Region(n) & mo).count() for n in nets]
    res["nets"] = len(nets)
    res["nets_without_pad"] = sum(1 for k in pads_per_net if k == 0)
    res["nets_with_several_pads"] = sum(1 for k in pads_per_net if k > 1)
    bad_heaters = 0
    for h in m1.each():
        touching = sum(1 for n in nets if not (kdb.Region(n) & kdb.Region(h)).is_empty())
        bad_heaters += touching != 2
    res["heaters"] = m1.count()
    res["heaters_not_on_two_nets"] = bad_heaters
    gc = region(ly, top, (81, 0)).merged()              # fibre targets of the grating couplers
    res["metal_near_grating"] = (m2 & gc.sized(um(25.0))).count()
    bb = top.dbbox()
    res["die_um"] = [round(bb.width(), 1), round(bb.height(), 1)]
    return res


def route_overlaps(component, routes):
    """For each route: silicon it shares with any other instance must lie at its end ports."""
    problems = []
    kcl = component.kcl
    li = kcl.layer(1, 0)

    def inst_region(inst):
        return kdb.Region(inst.cell.begin_shapes_rec(li)).transformed(inst.trans)

    insts = list(component.insts)
    key = lambda i: (i.cell_index, str(i.trans))  # noqa: E731
    regions = {key(i): inst_region(i) for i in insts}
    for name, r, p1, p2 in routes:
        mine = {key(i) for i in r.instances}
        rr = kdb.Region()
        for k in mine:
            rr += regions[k]
        others = kdb.Region()
        for k, reg in regions.items():
            if k not in mine:
                others += reg
        ov = (rr & others).merged()
        ends = [kdb.Point(int(p.center[0]), int(p.center[1])) for p in (p1, p2)]
        for poly in ov.each():
            c = poly.bbox().center()
            if min(c.distance(e) for e in ends) > 2000:     # 2 um from either end port
                problems.append((component.name, name, round(c.x / 1000, 1), round(c.y / 1000, 1)))
    return problems


if __name__ == "__main__":
    gds = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "full_adder_chip.gds")
    print(json.dumps(drc(gds), indent=1))
