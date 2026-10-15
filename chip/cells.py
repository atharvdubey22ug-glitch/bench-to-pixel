"""Building blocks for the full-adder chip, on the SiEPIC EBeam (Applied Nanotools) layer map.

Fabricated layers: Si 1/0 (220 nm strip waveguides), M1_heater 11/0 (TiW), M2_router 12/0,
M_Open 13/0 (pad openings), FloorPlan 99/0, text 10/0. The foundry components (grating coupler,
Y-branch, broadband directional coupler, terminator) are the official SiEPIC EBeam GDS cells,
with ports read from their PinRec (1/10) pins.
"""
import functools
import os
import glob

import gdsfactory as gf
import klayout.db as kdb

SI, M1, M2, MOPEN, FP, TXT, DEVREC = (1, 0), (11, 0), (12, 0), (13, 0), (99, 0), (10, 0), (68, 0)
RADIUS = 10.0
XS = gf.cross_section.cross_section(width=0.5, layer=SI, radius=RADIUS, radius_min=5.0)

_PDK_GDS = None


def pdk_gds_dir():
    """Folder with the official SiEPIC EBeam GDS cells (from the siepic_ebeam_pdk package)."""
    global _PDK_GDS
    if _PDK_GDS is None:
        env = os.environ.get("SIEPIC_EBEAM_GDS")
        cands = [env] if env else []
        cands += glob.glob(os.path.join(os.path.dirname(__file__), "pdk_gds"))
        try:
            import siepic_ebeam_pdk
            cands.append(os.path.join(os.path.dirname(siepic_ebeam_pdk.__file__), "gds", "EBeam"))
        except ImportError:
            pass
        for c in cands:
            if c and os.path.exists(os.path.join(c, "ebeam_y_1550.gds")):
                _PDK_GDS = c
                break
        if _PDK_GDS is None:
            raise FileNotFoundError("set SIEPIC_EBEAM_GDS to the siepic_ebeam_pdk gds/EBeam folder")
    return _PDK_GDS


@functools.cache
def foundry_cell(name):
    """Official cell with ports from its 1/10 pin paths (opt1..optN)."""
    path = os.path.join(pdk_gds_dir(), name + ".gds")
    ly = kdb.Layout()
    ly.read(path)
    top = ly.top_cell()
    pin = ly.layer(1, 10)
    texts, paths = {}, []
    for s in top.each_shape(pin):
        if s.is_text():
            texts[s.dtext.string] = (s.dtext.x, s.dtext.y)
        elif s.is_path():
            pts = list(s.dpath.each_point())
            paths.append((pts[0], pts[-1], s.dpath.width))
    imported = gf.import_gds(path)
    c = gf.Component(name + "_p")
    c << imported
    for nm, (tx, ty) in sorted(texts.items()):
        best = min(paths, key=lambda p: (0.5 * (p[0].x + p[1].x) - tx) ** 2 + (0.5 * (p[0].y + p[1].y) - ty) ** 2)
        a, b, w = best
        dx, dy = b.x - a.x, b.y - a.y
        ang = {(1, 0): 0, (-1, 0): 180, (0, 1): 90, (0, -1): 270}[(int(round(dx / abs(dx))) if dx else 0, int(round(dy / abs(dy))) if dy else 0)]
        cx, cy = 0.5 * (a.x + b.x), 0.5 * (a.y + b.y)
        c.add_port(name=nm, center=(cx, cy), width=w, orientation=ang, layer=SI, port_type="optical")
    return c


def gc():
    return foundry_cell("ebeam_gc_te1550")


def ybranch():
    return foundry_cell("ebeam_y_1550")


def bdc():
    return foundry_cell("ebeam_bdc_te1550")


def terminator():
    return foundry_cell("ebeam_terminator_te1550")


@gf.cell
def heater_wg(length=150.0, width=4.0, contact=20.0, overlap=10.0):
    """Straight waveguide under a TiW heater (M1) with M2 contact pads at both ends."""
    c = gf.Component()
    w = c << gf.components.straight(length=length + 2 * contact, cross_section=XS)
    w.dmovex(-contact)
    c.add_port("o1", port=w.ports["o1"])
    c.add_port("o2", port=w.ports["o2"])
    c.add_polygon([(0, -width / 2), (length, -width / 2), (length, width / 2), (0, width / 2)], layer=M1)
    # M1 runs under the contacts by `overlap` (>= 3 um rule)
    c.add_polygon([(-overlap, -width / 2), (0, -width / 2), (0, width / 2), (-overlap, width / 2)], layer=M1)
    c.add_polygon([(length, -width / 2), (length + overlap, -width / 2), (length + overlap, width / 2), (length, width / 2)], layer=M1)
    for x0, nm in ((-contact, "e1"), (length, "e2")):
        c.add_polygon([(x0, -contact / 2), (x0 + contact, -contact / 2), (x0 + contact, contact / 2), (x0, contact / 2)], layer=M2)
        c.add_port(nm, center=(x0 + contact / 2, 0), width=contact, orientation=90, layer=M2, port_type="electrical")
    return c


@gf.cell
def sbend(dx=40.0, dy=10.0):
    return gf.components.bend_s(size=(dx, dy), cross_section=XS)


@gf.cell
def mzi_switch(heater_length=150.0, half_sep=20.0, sbend_dx=40.0):
    """Thermo-optic Mach-Zehnder switch: BDC -> arms spread to +/-half_sep -> BDC.
    Ports: o1 = input (bdc1 opt1, left top); o2 = output (bdc2 opt3, right top);
    in2 (bdc1 opt2) and dump (bdc2 opt4) are terminated. Heater on the lower arm (bdc1 opt4 -> bdc2 opt2)."""
    c = gf.Component()
    b1 = c << bdc()
    up1 = c << sbend(sbend_dx, half_sep - 2.35)
    up1.connect("o1", b1.ports["opt3"])
    dn1 = c << sbend(sbend_dx, half_sep - 2.35)
    dn1.connect("o1", b1.ports["opt4"], mirror=True)
    top = c << gf.components.straight(length=heater_length + 40.0, cross_section=XS)
    top.connect("o1", up1.ports["o2"])
    h = c << heater_wg(length=heater_length)
    h.connect("o1", dn1.ports["o2"])
    up2 = c << sbend(sbend_dx, half_sep - 2.35)
    up2.connect("o1", top.ports["o2"], mirror=True)
    dn2 = c << sbend(sbend_dx, half_sep - 2.35)
    dn2.connect("o1", h.ports["o2"])
    b2 = c << bdc()
    b2.connect("opt1", up2.ports["o2"])
    for p, nm in ((b1.ports["opt2"], "t1"), (b2.ports["opt4"], "t2")):
        t = c << terminator()
        t.connect("opt1", p)
    c.add_port("o1", port=b1.ports["opt1"])
    c.add_port("o2", port=b2.ports["opt3"])
    c.add_port("e1", port=h.ports["e1"])
    c.add_port("e2", port=h.ports["e2"])
    c.info["arm_length"] = float(2 * up1.cell.info.get("length", 0) + heater_length + 40.0)
    return c


@gf.cell
def loop_mirror(radius=10.0, sbend_dx=30.0):
    """Sagnac loop mirror: Y-branch whose two outputs are joined by a waveguide loop."""
    c = gf.Component()
    y = c << ybranch()
    s_up = c << sbend(sbend_dx, radius - 2.75)
    s_up.connect("o1", y.ports["opt2"])
    s_dn = c << sbend(sbend_dx, radius - 2.75)
    s_dn.connect("o1", y.ports["opt3"], mirror=True)
    arc = c << gf.components.bend_circular(angle=180, cross_section=XS, radius=radius)
    arc.connect("o1", s_up.ports["o2"], mirror=True)
    a, b = arc.ports["o2"].dcenter, s_dn.ports["o2"].dcenter
    assert abs(a[0] - b[0]) < 1e-3 and abs(a[1] - b[1]) < 1e-3, (a, b)
    c.add_port("o1", port=y.ports["opt1"])
    c.info["loop_length"] = float(2 * sbend(sbend_dx, radius - 2.75).info.get("length", sbend_dx) + 3.141592653589793 * radius)
    return c


@gf.cell
def bond_pad(size=100.0, open_margin=5.0):
    c = gf.Component()
    c.add_polygon([(-size / 2, -size / 2), (size / 2, -size / 2), (size / 2, size / 2), (-size / 2, size / 2)], layer=M2)
    o = size / 2 - open_margin
    c.add_polygon([(-o, -o), (o, -o), (o, o), (-o, o)], layer=MOPEN)
    c.add_port("e1", center=(0, -size / 2), width=size, orientation=270, layer=M2, port_type="electrical")
    return c
