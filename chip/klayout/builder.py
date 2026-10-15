#!/usr/bin/env python3
"""
Layout builder for the optical full adder (paper Fig. 15, hybrid) in the SiEPIC EBeam process
(Applied Nanotools), from the EBeam library:

  GC_TE_1550_8degOxide_BB   grating couplers (fibre array, 127 um pitch)
  ebeam_y_1550              Y-branches (S1, taps, loop mirrors)
  ebeam_bdc_te1550          broadband directional couplers (Michelson XORs, Mach-Zehnder switches)
  ebeam_terminator_te1550   terminators on unused coupler ports
  wg_heater (PCell)         TiW heaters (switch arms, Michelson bias)
  Waveguide (PCell)         every waveguide, 'Strip TE 1550 nm, w=500 nm'
  ebeam_BondPad (EBeam_Beta) probe pads

Builder(top).full_adder(...) draws one full adder (two automated-test circuits) at an offset.
full_adder_siepic.py places one; alu4_siepic.py places four (a 4-bit ripple-carry ALU).
Builder.finish() adds the floor plan, writes .gds/.oas, runs SiEPIC verification, and writes the
SiEPIC-extracted netlist (per bit) used by the circuit simulation.
"""
import json
import math
import os

import pya
from pya import DPoint, DPath, DCplxTrans

import SiEPIC  # noqa: F401
import siepic_ebeam_pdk  # noqa: F401  (registers the EBeam technology and libraries)
from SiEPIC.utils.layout import new_layout
from SiEPIC.extend import to_itype  # noqa: F401

USER = "atharvdubey"
WG = "Strip TE 1550 nm, w=500 nm"
HEATER = 120.0
D = 240.6           # Mach-Zehnder switch: coupler-to-coupler distance


def P(T, x, y):
    return T * DPoint(x, y)


class Builder:
    def __init__(self, top):
        self.top = top
        self.cell, self.ly = new_layout("EBeam", top, GUI=False, overwrite=True)
        ly = self.ly
        self.dbu = ly.dbu
        self.L_M2 = ly.layer(ly.TECHNOLOGY["M2_router"])
        self.L_TXT = ly.layer(ly.TECHNOLOGY["Text"])
        self.C_GC = ly.create_cell2("GC_TE_1550_8degOxide_BB", "EBeam")
        self.C_Y = ly.create_cell2("ebeam_y_1550", "EBeam")
        self.C_BDC = ly.create_cell2("ebeam_bdc_te1550", "EBeam")
        self.C_TERM = ly.create_cell2("ebeam_terminator_te1550", "EBeam")
        self.C_PAD = ly.create_cell2("ebeam_BondPad", "EBeam_Beta")
        self.C_HEAT = ly.create_cell("wg_heater", "EBeam", {"length": HEATER, "mh_width": 4, "waveguide_type": WG})
        self.SHIFT = DCplxTrans()
        self.lengths, self.heaters, self.gc_roles = {}, {}, {}
        self.bits = []          # (bit, absolute DBox region, heater-name prefix, design->GDS transform)
        self.blocks = []
        self.opt_in = []
        self.heaters_per_bit = 10

    # ------------------------------------------------------------ primitives (design coordinates)
    def place(self, c, T, x=0.0, y=0.0, rot=0, mirror=False):
        t = self.SHIFT * T * DCplxTrans(1, rot, mirror, x, y)
        return self.cell.insert(pya.DCellInstArray(c.cell_index(), t))

    def pinp(self, inst, name):
        """Pin centre in design coordinates (micron, before SHIFT)."""
        p = inst.find_pin(name)
        return self.SHIFT.inverted() * DPoint(p.center.x * self.dbu, p.center.y * self.dbu)

    def wg(self, key, pts_design):
        """Waveguide PCell along a Manhattan path given in design coordinates; returns its length."""
        pts = [self.SHIFT * p for p in pts_design]
        c = self.ly.create_cell("Waveguide", "EBeam", {"path": DPath(pts, 0.5), "waveguide_type": WG})
        self.cell.insert(pya.CellInstArray(c.cell_index(), pya.Trans()))
        L = float(c.display_title().split("_")[-1])
        key = self.pre + key
        self.lengths[key] = self.lengths.get(key, 0.0) + L
        return L

    def wire(self, pts, width=10.0):
        """Manhattan M2 wire in design coordinates."""
        for a, b in zip(pts[:-1], pts[1:]):
            a, b = self.SHIFT * a, self.SHIFT * b
            if abs(a.x - b.x) < 1e-9:
                box = pya.DBox(a.x - width / 2, min(a.y, b.y) - width / 2, a.x + width / 2, max(a.y, b.y) + width / 2)
            else:
                box = pya.DBox(min(a.x, b.x) - width / 2, a.y - width / 2, max(a.x, b.x) + width / 2, a.y + width / 2)
            self.cell.shapes(self.L_M2).insert(box)

    def label(self, text, p, size=5.0):
        q = self.SHIFT * p
        t = pya.Text(text, pya.Trans(pya.Trans.R0, int(round(q.x / self.dbu)), int(round(q.y / self.dbu))))
        self.cell.shapes(self.L_TXT).insert(t).text_size = size / self.dbu

    # ------------------------------------------------------------ Mach-Zehnder switch
    # local frame: input bdc1.opt1 at (-35.45, 2.35) facing west, output bdc2.opt3 facing east;
    # heater on the lower arm (y = -20); arms are mirror images, so equal in length.
    def switch(self, name, T):
        b1 = self.place(self.C_BDC, T, 0, 0)
        b2 = self.place(self.C_BDC, T, D, 0)
        h = self.place(self.C_HEAT, T, 60.0, -20.0)
        t1 = self.place(self.C_TERM, T, -35.45, -2.35)
        t2 = self.place(self.C_TERM, T, D + 35.3, -2.35, rot=180)
        self.wg(f"{name}_arm_top", [P(T, 35.3, 2.35), P(T, 45.3, 2.35), P(T, 45.3, 20), P(T, D - 45.45, 20), P(T, D - 45.45, 2.35), P(T, D - 35.45, 2.35)])
        self.wg(f"{name}_arm_heater", [P(T, 35.3, -2.35), P(T, 45.3, -2.35), P(T, 45.3, -20), P(T, 60, -20)])
        self.wg(f"{name}_arm_heater", [P(T, 60 + HEATER, -20), P(T, D - 45.45, -20), P(T, D - 45.45, -2.35), P(T, D - 35.45, -2.35)])
        self.lengths[self.pre + f"{name}_arm_heater"] += HEATER
        e1, e2 = self.pinp(h, "elec1"), self.pinp(h, "elec2")
        self.heaters[self.pre + name] = sorted([(e1.x, e1.y), (e2.x, e2.y)])
        return dict(inp=P(T, -35.45, 2.35), out=P(T, D + 35.3, 2.35), insts=(b1, b2, h, t1, t2))

    def loop_mirror(self, name, T):
        """Y-branch at the local origin (input opt1 at (-7.4, 0) facing west) with a rectangular loop."""
        self.place(self.C_Y, T, 0, 0)
        pts = [(7.4, 2.75), (20, 2.75), (20, 15), (45, 15), (45, -15), (20, -15), (20, -2.75), (7.4, -2.75)]
        self.wg(f"{name}_loop", [P(T, *p) for p in pts])
        return P(T, -7.4, 0)

    # ------------------------------------------------------------ half block
    # Each block is one automated-test circuit (SiEPIC Phot1x design-for-test rules: one laser
    # coupler with at most one detector coupler above it and two below, 127 um pitch):
    #     y = 381  XOR output            (detector above the laser)
    #     y = 254  LASER in              (opt_in label)
    #     y = 127  AND branch n1         (detector)
    #     y =   0  AND branch n2         (detector)
    # Block 1 (inputs A, B) is the paper's half adder; block 2 (inputs Cin, X) is the second half
    # adder of the textbook full adder (Fig. 12); the carries are added on the photodiodes.
    def block(self, h, T, n1, n2, opt_name, pad_name):
        place, wg, label, switch = self.place, self.wg, self.label, self.switch
        heaters, lengths, pre = self.heaters, self.lengths, self.pre
        C_GC, C_Y, C_BDC, C_HEAT, C_PAD = self.C_GC, self.C_Y, self.C_BDC, self.C_HEAT, self.C_PAD
        for r in (0.0, 127.0, 254.0, 381.0):
            place(C_GC, T, 0, r)
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
        e1, e2 = self.pinp(hb, "elec1"), self.pinp(hb, "elec2")
        heaters[pre + f"bias{h}"] = sorted([(e1.x, e1.y), (e2.x, e2.y)])
        lengths[pre + f"arm_a{h}"] += HEATER
        # arm b: switch east of the bias heater, then loop mirror
        xb_in, yb2 = xo + HEATER + 70.0, yb + 60.0
        swb = switch(f"{n2}_x{h}", T * DCplxTrans(1, 0, True, xb_in + 35.45, yb2 + 2.35))
        wg(f"arm_b{h}", [P(T, *p_b), P(T, p_b[0] + 26, p_b[1]), P(T, p_b[0] + 26, yb2), P(T, xb_in, yb2)])
        xbo = (T.inverted() * swb["out"]).x
        self.loop_mirror(f"lmb{h}", T * DCplxTrans(1, 0, False, xbo + 7.4, yb2))
        extra = max(lengths[pre + f"arm_b{h}"] - lengths[pre + f"arm_a{h}"], 20.0)
        wg(f"arm_a{h}", [P(T, xo + HEATER, ya2), P(T, xo + HEATER + extra, ya2)])
        self.loop_mirror(f"lma{h}", T * DCplxTrans(1, 0, False, xo + HEATER + extra + 7.4, ya2))
        # labels
        opt = f"opt_in_TE_1550_device_{USER}_{opt_name}"
        label(opt, T * DPoint(0, 254.0))
        self.opt_in.append(opt)
        roles = {381.0: "xor" if h == 1 else "sum", 254.0: f"laser{h}", 127.0: f"c_{'a' if h == 1 else 'c'}", 0.0: f"c_{'b' if h == 1 else 'd'}"}
        for r, nm in roles.items():
            self.gc_roles[(self.bit, nm)] = self.SHIFT * T * DPoint(0, r)
        # ---- pads and wires (chain: pad k joins heater k's right contact and heater k+1's left)
        y_pad = 381.0 + 170.0
        pads_x = [150.0 + 125.0 * k for k in range(6)]
        pads = []
        for k, x in enumerate(pads_x):
            place(C_PAD, T, x, y_pad + 50.0)
            pads.append(pad_name(h, k))
            label(f"elec_{USER}_{pads[-1]}", T * DPoint(x, y_pad + 50.0))
        order = [f"{n2}_and", f"{n1}_and", f"{n1}_x{h}", f"bias{h}", f"{n2}_x{h}"]
        loc = [[T.inverted() * DPoint(*p) for p in heaters[pre + nm]] for nm in order]
        (L1, R1), (L2, R2), (L3, R3), (L4, R4), (L5, R5) = [sorted(v, key=lambda q: q.x) for v in loc]
        ych = [0.0, y_pad - 130.0] + [y_pad - 110.0 + 20.0 * (k - 2) for k in range(2, 6)]
        ych[0] = ych[1] + 2.0
        ych[2] = y_pad - 150.0          # below pad 1's channel, so heater 2's right wire clears pad 1's wire

        def W(pts):
            self.wire([T * DPoint(*p) if isinstance(p, tuple) else T * p for p in pts])

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
        blk = dict(bit=self.bit, block=h, opt_in=opt, pads=pads,
                   heaters=[dict(name=nm, between=[pads[k], pads[k + 1]]) for k, nm in enumerate(order)])
        self.blocks.append(blk)
        return blk

    # ------------------------------------------------------------ one full adder
    def full_adder(self, bit, dx, dy, opt_name, pad_name, pitch, loopback=False, prefix=True):
        """One full adder; design origin at GDS (120 + dx, 860 + dy). opt_name(h) and pad_name(h, k)
        give the test-circuit and pad label names. pitch = (w, h) of the cell reserved for this bit."""
        self.bit = bit
        self.pre = f"b{bit}:" if prefix else ""
        self.SHIFT = DCplxTrans(1, 0, False, 120.0 + dx, 860.0 + dy)
        ID = DCplxTrans()
        self.block(1, ID, "A", "B", opt_name(1), pad_name)
        self.block(2, DCplxTrans(1, 0, False, 0, -762.0), "C", "X", opt_name(2), pad_name)
        if loopback:            # for fibre-array alignment
            self.place(self.C_GC, ID, 0, 508.0)
            self.place(self.C_GC, ID, 0, 635.0)
            self.wg("loopback", [DPoint(0, 508), DPoint(20, 508), DPoint(20, 635), DPoint(0, 635)])
            self.label(f"opt_in_TE_1550_device_{USER}_Loopback", DPoint(0, 635))
            self.gc_roles[(bit, "loopback_in")] = self.SHIFT * DPoint(0, 635)
            self.gc_roles[(bit, "loopback_out")] = self.SHIFT * DPoint(0, 508)
            self.opt_in.append(f"opt_in_TE_1550_device_{USER}_Loopback")
        self.bits.append((bit, pya.DBox(dx, dy, dx + pitch[0], dy + pitch[1]), self.pre, self.SHIFT))

    # ------------------------------------------------------------ floor plan, files, checks
    def finish(self, here, dft_at=(80.0, 1560.0), write_spice=True):
        cell, ly, dbu = self.cell, self.ly, self.dbu
        top = self.top
        self.SHIFT = DCplxTrans()
        self.pre = ""
        self.label("DFT=DFT_Phot1x", DPoint(*dft_at), 10)
        bb = cell.dbbox()
        fp = ly.create_cell("FloorPlan")
        fp.shapes(ly.layer(ly.TECHNOLOGY["FloorPlan"])).insert(pya.DBox(0, 0, math.ceil(bb.right + 40), math.ceil(bb.top + 40)))
        cell.insert(pya.CellInstArray(fp.cell_index(), pya.Trans()))
        assert bb.left > 40 and bb.bottom > 40, bb

        gds = os.path.join(here, top + ".gds")
        ly.write(gds)
        from SiEPIC.scripts import export_layout
        export_layout(cell, here, top, format="oas", screenshot=False)
        from SiEPIC.verification import layout_check
        rdb = os.path.join(here, top + ".lyrdb")
        n_err = layout_check(cell=cell, verbose=False, GUI=False, file_rdb=rdb)
        spice = {}
        if write_spice:
            for opt in self.opt_in:
                if opt.endswith("Loopback"):
                    continue
                sp = cell.spice_netlist_export(verbose=False, opt_in_selection_text=[opt])
                spice[opt] = sp[0] if isinstance(sp, tuple) else str(sp)

        # ---- extracted netlist (SiEPIC net extraction: pins that touch), split per bit
        nets, comps = cell.identify_nets()

        def apt(p):
            return DPoint(p.x * dbu, p.y * dbu)

        def bit_of(q):
            for b, box, _, _ in self.bits:
                if box.contains(q):
                    return b
            raise ValueError(q)

        ext = []
        # heater midpoints, stored in each bit's design coordinates -> absolute
        abs_heater = {}
        for b, box, pre, sh in self.bits:
            for nm, pts in self.heaters.items():
                if nm.startswith(pre) and (pre or ":" not in nm):
                    abs_heater[(b, nm[len(pre):])] = sh * DPoint(0.5 * (pts[0][0] + pts[1][0]), pts[0][1])
        for c in comps:
            name = c.basic_name or c.component
            pins = {p.pin_name: p.net.idx for p in c.pins if p.type == 1 or str(p.pin_name).startswith("opt")}
            cc = apt(c.center)
            b = bit_of(cc)
            item = dict(idx=c.idx, kind=name, bit=b, pins=pins, center=[round(cc.x, 3), round(cc.y, 3)])
            prm = dict(kv.split("=", 1) for kv in (c.params or "").split() if "=" in kv)
            if "wg_length" in prm:
                v = prm["wg_length"].rstrip("u")
                item["length_um"] = float(v) if prm["wg_length"].endswith("u") else float(v) * 1e6
            if name == "wg_heater":
                cand = {k[1]: p for k, p in abs_heater.items() if k[0] == b}
                item["role"] = min(cand, key=lambda k: (cand[k].x - cc.x) ** 2 + (cand[k].y - cc.y) ** 2)
            if name.startswith("GC_"):
                for (bb_, role), p in self.gc_roles.items():
                    if bb_ == b and abs(p.y - cc.y) < 20 and abs(p.x - cc.x) < 60:
                        item["role"] = role
                if "role" not in item:
                    item["role"] = f"gc_{int(round(cc.y))}"
            ext.append(item)
        for b, _, _, _ in self.bits:
            roles = [e.get("role") for e in ext if e["kind"] == "wg_heater" and e["bit"] == b]
            assert len(set(roles)) == len(roles) == self.heaters_per_bit, (b, roles)
        # ---- independent checks
        li = ly.layer(1, 0)
        insts = [i for i in cell.each_inst() if i.cell.name != "FloorPlan"]
        regs = [pya.Region(i.cell.begin_shapes_rec(li)).transformed(i.cplx_trans) for i in insts]
        all_pins = [p.center for c in comps for p in c.pins]
        crossings = []
        boxes = [r.bbox() for r in regs]
        order = sorted(range(len(insts)), key=lambda k: boxes[k].left)
        for a_i, i in enumerate(order):          # sweep in x: only overlapping boxes are compared
            for j in order[a_i + 1:]:
                if boxes[j].left > boxes[i].right:
                    break
                if not boxes[i].overlaps(boxes[j]):
                    continue
                ov = (regs[i] & regs[j]).merged()
                for poly in ov.each():
                    cc = poly.bbox().center()
                    if min((cc - p).length() for p in all_pins) > 2000:
                        q = apt(cc)
                        crossings.append([round(q.x, 1), round(q.y, 1)])
        cnt = {}
        for e in ext:
            for nid in e["pins"].values():
                cnt[nid] = cnt.get(nid, 0) + 1
        dangling = [(e["kind"], e["center"]) for e in ext for nid in e["pins"].values() if cnt[nid] < 2]
        fpb = fp.dbbox()
        return dict(gds=gds, rdb=rdb, spice=spice, ext=ext, n_err=n_err, crossings=crossings,
                    dangling=dangling, die=[fpb.width(), fpb.height()])
