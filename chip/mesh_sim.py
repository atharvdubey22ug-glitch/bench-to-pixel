#!/usr/bin/env python3
"""
Optical matrix processor (klayout/EBeam_heaters_atharvdubey_Mesh8): circuit simulation, calibration and
a two-layer digit classifier run on the simulated chip.

The chip. Laser -> 1x8 Y-branch tree -> 8 push-pull encoders (input values x_j in [-1, 1]) -> 8x8
Clements mesh of 28 Mach-Zehnder interferometers (56 heaters) -> outputs 7, 0, 1 on the three detector
grating couplers the test rules allow. With the mesh set, detector d reads |m_d . x|^2, where m_d is a
row of the mesh's unitary matrix; three rows are read at a time ("a page").

Circuit model. The circuit is read from the SiEPIC extraction of the layout (every component, net and
drawn waveguide). Each component is the PDK compact model at the process corner (grating couplers,
Y-branches, broadband couplers, strip waveguides with loss). The input section (laser coupler, 1x8
tree, encoders) is solved with all its reflections (node-and-link scattering; the PDK Y-branch
reflects 0.6% into its branch ports, so the encoder arms are weak cavities); the mesh and output
routes are propagated forward (Circuit). full_check() compares that with the full SAX scattering
solution, every reflection included, on random heater settings: 0.4% of the largest output.

A simulated die adds:
  * heater efficiency pi per 25 mW, spread +-10% (1 sigma);
  * a random static phase on every waveguide (uniform 0..2 pi): every interferometer, encoder and
    channel starts at an unknown phase;
  * thermal crosstalk from every heater onto every waveguide near it, integrated along the drawn paths:
    a parallel waveguide at distance d picks up 5% exp(-(d - 100 um) / 60 um) of the heater's phase,
    the same law as the neuron chip;
  * the nine process corners (silicon 210/220/230 nm x width 480/500/520 nm), which move the couplers'
    split from 44/56 to 53/47;
  * 16-bit heater drivers (0 to 80 mW) and detector reading noise of 0.2% of the reading when all of
    one input's light reaches one detector.

Calibration, from the three detectors only (calibrate_encoders(), Page):
  1. encoders: a 2-D sweep of each encoder's two heaters, fitted with the push-pull model, gives the
     drive for any signed input at a fixed output phase (calibrate_encoder); each zero is then refined
     with the other encoders at zero (refine_nulls), and the phase-sum shift that turns an encoder's
     output a quarter period is measured against a partner (calibrate_quadrature);
  2. mesh heater periods: each heater swept over the driver's range (mesh_periods);
  3. pages: for a wanted set of three rows, the mesh's 56 heaters are found by Levenberg-Marquardt on
     measured data only. Detector d reads x^T Re(H_d) x for real inputs, H_d = conj(m_d) m_d^T. H_d is
     measured with 36 inputs (each alone and every pair) plus the 28 pairs with one input turned a
     quarter period (imaginary parts); the Jacobian is measured too (each heater nudged in turn), so
     crosstalk and coupler errors are part of what is fitted. Rows are added one at a time with the
     complex rows as target, then Re H alone is matched; a fit that stays above the noise is shaken
     and refitted, or restarted.
  The layer's outputs are then read as R_d / (sum of the single-input readings): |t_d . x|^2 for unit
  rows, whatever each encoder's and detector's own loss.

Network (mesh_net.py): MNIST digits reduced to their first 7 principal components (computed in
electronics), plus a constant bias input, x = [7 features, 1]. Layer 1: seven rows in pages of 3, 3
and 1; the detected intensities pass through a clipped ramp in electronics, z = clip(a h + b, 0, 1)
(7 scales, 7 offsets), and with a bias input form layer 2's input. Layer 2: ten rows (one per digit),
pages of 3, 3, 3 and 1; the largest of the ten detected intensities is the answer. The network is
trained in software with exactly this structure (each page the first rows of a unitary), then run
on the simulated chips: 7 programmed pages per die, 10,000 test digits.
"""
import json
import math
import os
import sys
import time
import warnings

import numpy as np

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import full_adder_sim as F  # noqa: E402
from parity_sim import _fit, dac, DAC_MAX  # noqa: E402

EXTRACT = os.path.join(HERE, "klayout", "mesh_extracted_netlist.json")
N = 8
DETS = ["o7", "o0", "o1"]                 # detector couplers: mesh outputs 7, 0, 1
DET_CH = [7, 0, 1]
P_PI, ETA_SPREAD, MEAS_NOISE = 25e-3, 0.10, 2e-3
XT0, XT_D0, XT_L, XT_REACH = 0.05, 100.0, 60.0, 400.0
ACCEPT = 1e-3                             # measured misfit per row below which a page run is accepted
MAX_STEP = 12e-3                          # largest heater change per programming step (W)
_EXT = json.load(open(EXTRACT))
MZIS = _EXT["mzis"]
ENC_H = [f"E{i}{a}" for i in range(N) for a in "AB"]
MESH_H = [m[k] for m in MZIS for k in ("phi", "theta")]
HEATERS = ENC_H + MESH_H                  # 16 + 56
SIDES = {"Waveguide": (("opt1",), ("opt2",)), "wg_heater": (("opt1",), ("opt2",)),
         "ebeam_y_1550": (("opt1",), ("opt2", "opt3")), "ebeam_bdc_te1550": (("opt1", "opt2"), ("opt3", "opt4"))}


def sdict(kind, corner, wl, length=None):
    """Forward S-parameters of one component at one wavelength: {(pin_in, pin_out): complex}."""
    th, w = corner
    wl = np.atleast_1d(wl)
    if kind in ("Waveguide", "wg_heater"):
        s = F.MODELS["wg"](wl=wl, length=length, loss=F.LOSS, width=w, height=th)
        t = np.asarray(s[("o0", "o1")])
        return {("opt1", "opt2"): t, ("opt2", "opt1"): t}
    model, mp = F.PORTMAP[kind]
    st = {"thickness": th} if model == "gc" else {"thickness": th, "width": w}
    s = F.MODELS[model](wl=wl, **st)
    inv = {v: k for k, v in mp.items()}
    out = {}
    for (a, b), v in s.items():
        if a in inv and b in inv:
            out[(inv[a], inv[b])] = np.asarray(v)
    if model == "gc":
        out = {("fibre", "opt1"): np.asarray(s[("o0", "o1")]), ("opt1", "fibre"): np.asarray(s[("o1", "o0")])}
    return out


class Circuit:
    """The drawn circuit at one corner, propagated forward from the laser coupler.

    run(phase) takes the extra phase of every waveguide and heater component (B x n_wg array, in the
    order of self.wg) and returns the fields at the three detectors (B x 3) and at all eight mesh
    outputs (B x 8, before the output routes)."""

    def __init__(self, corner=F.NOMINAL, wl=1.55, comps=None):
        self.corner, self.wl = corner, wl
        comps = comps or _EXT["components"]
        self.comps = comps
        net_pins = {}
        for c in comps:
            for p, n in c["pins"].items():
                net_pins.setdefault(n, []).append((c["idx"], p))
        byidx = {c["idx"]: c for c in comps}
        self.wg = [c["idx"] for c in comps if c["kind"] in ("Waveguide", "wg_heater")]
        self.wg_pos = {k: j for j, k in enumerate(self.wg)}
        self.wg_role = {c["role"]: self.wg_pos[c["idx"]] for c in comps if c["kind"] == "wg_heater"}
        gc = {c["role"]: c for c in comps if c["kind"].startswith("GC_")}
        cache = {}

        def S(c):
            key = (c["kind"], c.get("length_um"))
            if key not in cache:
                cache[key] = {k: complex(np.ravel(v)[0]) for k, v in sdict(c["kind"], corner, wl, c.get("length_um")).items()}
            return cache[key]

        nets = {}                                  # net -> slot in the field array
        ops = []
        src = gc["laser"]
        n0 = src["pins"]["opt1"]
        nets[n0] = 0
        self.t_in = S(src)[("fibre", "opt1")]
        fired = {src["idx"]}
        frontier = [n0]
        pending = {}
        while frontier:
            n = frontier.pop()
            for (ci, pin) in net_pins[n]:
                if ci in fired:
                    continue
                c = byidx[ci]
                k = c["kind"]
                if k.startswith("GC_") or k.startswith("ebeam_terminator"):
                    fired.add(ci)
                    continue
                sides = SIDES[k]
                side_in = 0 if pin in sides[0] else 1
                ins, outs = sides[side_in], sides[1 - side_in]
                have = pending.setdefault(ci, set())
                have.add(pin)
                if not all(c["pins"][p] in nets for p in ins if p in c["pins"]):
                    continue
                fired.add(ci)
                s = S(c)
                M = np.array([[s[(a, b)] for a in ins] for b in outs])
                out_nets = []
                for b in outs:
                    m = c["pins"][b]
                    nets[m] = len(nets)
                    out_nets.append(nets[m])
                    frontier.append(m)
                wgk = self.wg_pos.get(ci, -1)
                ops.append((np.array([nets[c["pins"][a]] for a in ins]), np.array(out_nets), M, wgk, ci))
        self.n_nets = len(nets)
        self._input_section(comps, byidx, net_pins, nets, S, gc["laser"])
        self.ops = [op for op in ops if op[4] not in self.in_comps]
        self.det_net = [nets[gc[d]["pins"]["opt1"]] for d in DETS]
        self.t_out = [S(gc[d])[("opt1", "fibre")] for d in DETS]
        # the eight mesh outputs: the nets entering the output routes and the terminators
        self.out_net = {}
        for c in comps:
            key = c.get("key", "")
            if c["kind"] == "Waveguide" and (key.startswith("route_o") or key.endswith("_term")):
                ch = int(key[7:]) if key.startswith("route_o") else int(key[3:key.index("_")])
                self.out_net[ch] = min(nets.get(c["pins"]["opt1"], 1e9), nets.get(c["pins"]["opt2"], 1e9))
        assert sorted(self.out_net) == list(range(N)), self.out_net
        assert len(fired) == len(comps), (len(fired), len(comps))

    def _input_section(self, comps, byidx, net_pins, nets, S, gc_laser):
        """Laser coupler, 1x8 tree and encoders, solved with all their reflections: the PDK Y-branch
        reflects 0.6% of the light back into its branch ports and couples 0.3% across, so the encoder arms
        are weak cavities and the encoders talk to each other through the tree. Node-and-link scattering
        problem: nodes are the coupler, the tree's Y-branches and the encoders (each a 2-port solved with
        its own internal reflections); links are chains of waveguides between them."""
        keys = {c["idx"]: c.get("key", "") for c in comps}
        in_wg = {ci for ci, k in keys.items() if k == "laser_in" or k.startswith("tree") or (k.startswith("ch") and k.endswith("_in"))}
        enc_wg = {ci for ci, k in keys.items() if k.startswith("enc")}
        ys = {c["idx"] for c in comps if c["kind"] == "ebeam_y_1550"}
        # encoder Y-branches: those touching an encoder arm
        enc_y = {cj for ci in enc_wg for n in byidx[ci]["pins"].values() for (cj, _) in net_pins[n] if cj in ys}
        tree_y = {cj for ci in in_wg for n in byidx[ci]["pins"].values() for (cj, _) in net_pins[n] if cj in ys} - enc_y
        self.in_comps = in_wg | enc_wg | enc_y | tree_y | {gc_laser["idx"]}

        def other(ci, pin):
            n = byidx[ci]["pins"][pin]
            return [(cj, q) for (cj, q) in net_pins[n] if cj != ci]

        # encoders: Y1 (fed from a channel) and Y2, and each arm's chain
        encs = []
        for i in range(N):
            arm_idx = {a: [ci for ci in enc_wg if keys[ci] == f"enc{i}_{a}"] for a in "AB"}
            yy = {cj for a in "AB" for ci in arm_idx[a] for n in byidx[ci]["pins"].values() for (cj, _) in net_pins[n] if cj in enc_y}
            y1 = [cj for cj in yy if any(byidx[ck]["kind"] == "Waveguide" and keys[ck] == f"ch{i}_in" for ck, _ in other(cj, "opt1"))][0]
            y2 = [cj for cj in yy if cj != y1][0]
            arms = []
            for q in ("opt2", "opt3"):
                chain, (ci, pin) = [], other(y1, q)[0]
                while byidx[ci]["kind"] != "ebeam_y_1550":
                    chain.append(ci)
                    pin2 = "opt2" if pin == "opt1" else "opt1"
                    ci, pin = other(ci, pin2)[0]
                assert ci == y2
                arms.append((q, pin, [(S(byidx[c])[("opt1", "opt2")], self.wg_pos[c]) for c in chain]))
            S1, S2 = S(byidx[y1]), S(byidx[y2])
            q1, q2 = [a[0] for a in arms], [a[1] for a in arms]
            encs.append(dict(y1=y1, out=nets[byidx[y2]["pins"]["opt1"]],
                             s11=S1[("opt1", "opt1")],
                             s1=np.array([S1[("opt1", q)] for q in q1]),          # into the arms
                             s1r=np.array([S1[(q, "opt1")] for q in q1]),         # back out of Y1
                             R1=np.array([[S1[(a, b)] for a in q1] for b in q1]),  # Y1 seen from the arms
                             R2=np.array([[S2[(a, b)] for a in q2] for b in q2]),  # Y2 seen from the arms
                             s2=np.array([S2[(q, "opt1")] for q in q2]),
                             t=np.array([np.prod([t for t, _ in a[2]]) for a in arms]),
                             wgk=[[w for _, w in a[2]] for a in arms]))
        # nodes: ("gc", idx), ("y", idx), ("enc", i); ports per node
        node_ports = {("gc", gc_laser["idx"]): ["opt1"]}
        for cj in tree_y:
            node_ports[("y", cj)] = ["opt1", "opt2", "opt3"]
        for i, e in enumerate(encs):
            node_ports[("enc", i)] = ["opt1"]
        node_of = {gc_laser["idx"]: ("gc", gc_laser["idx"])}
        node_of.update({cj: ("y", cj) for cj in tree_y})
        node_of.update({e["y1"]: ("enc", i) for i, e in enumerate(encs)})
        links, seen = [], set()
        for nd, ports in node_ports.items():
            ci0 = nd[1] if nd[0] != "enc" else encs[nd[1]]["y1"]
            for p in ports:
                if (nd, p) in seen:
                    continue
                chain, (ci, pin) = [], other(ci0, p)[0]
                while ci in in_wg:
                    chain.append(ci)
                    pin2 = "opt2" if pin == "opt1" else "opt1"
                    ci, pin = other(ci, pin2)[0]
                end = (node_of[ci], pin)
                seen.update({(nd, p), end})
                links.append(((nd, p), end, [(S(byidx[c])[("opt1", "opt2")], self.wg_pos[c]) for c in chain]))
        # unknowns: wave leaving each end of each link (2 per link)
        ends = {}
        for k, (e0, e1, _) in enumerate(links):
            ends[e0], ends[e1] = 2 * k, 2 * k + 1
        nU = 2 * len(links)
        Sn = np.zeros((nU, nU), complex)           # outgoing[p] += S[q->p] * incoming[q], static nodes
        for nd, ports in node_ports.items():
            if nd[0] == "enc":
                continue
            s = S(byidx[nd[1]])
            for p in ports:
                for q in ports:
                    Sn[ends[(nd, p)], ends[(nd, q)]] = s.get((q, p), 0.0) if nd[0] == "y" else 0.0
        # grating coupler: waveguide-side reflection from the PDK model
        g = F.MODELS["gc"](wl=np.atleast_1d(self.wl), thickness=self.corner[0])
        r_gc = complex(np.ravel(g[("o1", "o1")])[0])
        Sn[ends[(("gc", gc_laser["idx"]), "opt1")], ends[(("gc", gc_laser["idx"]), "opt1")]] = r_gc
        src = np.zeros(nU, complex)
        src[ends[(("gc", gc_laser["idx"]), "opt1")]] = self.t_in
        self.inp = dict(links=links, ends=ends, nU=nU, Sn=Sn, src=src, encs=encs,
                        enc_end=[ends[(("enc", i), "opt1")] for i in range(N)],
                        partner=np.array([ends[e1] if j % 2 == 0 else ends[e0] for k, (e0, e1, _) in enumerate(links) for j in (0, 1)]),
                        link_t=np.array([np.prod([t for t, _ in L[2]]) if L[2] else 1.0 for L in links]),
                        link_wg=[[w for _, w in L[2]] for L in links])

    def _solve_inputs(self, phase):
        """Fields at the eight encoder outputs (B x 8), from the full scattering solution of the input section."""
        d = self.inp
        B = phase.shape[0]
        tau = np.stack([d["link_t"][k] * np.exp(1j * phase[:, ws].sum(1)) if ws else np.full(B, d["link_t"][k], complex)
                        for k, ws in enumerate(d["link_wg"])], 1)                        # (B, links)
        tau_u = np.repeat(tau, 2, axis=1)                                                    # per unknown (its link)
        # encoders: input reflection and transmission per sample
        r_enc, t_enc = [], []
        for e in d["encs"]:
            u = np.stack([e["t"][k] * np.exp(1j * phase[:, e["wgk"][k]].sum(1)) for k in range(2)], 1)
            M = np.eye(2)[None] - (e["R1"][None] * u[:, None, :]) @ (e["R2"][None] * u[:, None, :])
            fw = np.linalg.solve(M, np.broadcast_to(e["s1"][None, :, None], (B, 2, 1)))[..., 0]      # per unit input
            bw = ((e["R2"][None] * u[:, None, :]) @ fw[..., None])[..., 0]
            r_enc.append(e["s11"] + (e["s1r"][None] * u * bw).sum(1))
            t_enc.append((e["s2"][None] * u * fw).sum(1))
        Sn = np.broadcast_to(d["Sn"], (B,) + d["Sn"].shape).copy()
        for i, ee in enumerate(d["enc_end"]):
            Sn[:, ee, ee] = r_enc[i]
        # incoming at end j = tau * outgoing at the partner end of the same link
        P = np.zeros((d["nU"], d["nU"]))
        P[np.arange(d["nU"]), d["partner"]] = 1.0
        A = np.eye(d["nU"])[None] - Sn @ (tau_u[:, :, None] * P[None])
        o = np.linalg.solve(A, np.broadcast_to(d["src"][None, :, None], (B, d["nU"], 1)))[..., 0]
        inc = tau_u * o[:, d["partner"]]
        return np.stack([t_enc[i] * inc[:, ee] for i, ee in enumerate(d["enc_end"])], 1)

    def run(self, phase):
        phase = np.atleast_2d(phase)
        B = phase.shape[0]
        f = np.zeros((self.n_nets, B), complex)
        enc = self._solve_inputs(phase)
        for i, e in enumerate(self.inp["encs"]):
            f[e["out"]] = enc[:, i]
        e = np.exp(1j * phase)
        for op in self.ops:
            ins, outs, M, wgk, _ = op
            v = M @ f[ins]
            if wgk >= 0:
                v = v * e[:, wgk]
            f[outs] = v
        det = np.stack([f[n] * t for n, t in zip(self.det_net, self.t_out)], 1)
        mesh_out = np.stack([f[self.out_net[k]] for k in range(N)], 1)
        return det, mesh_out

    def _split(self):
        """The mesh section after the eight converger waveguides is linear in their output fields and
        only mesh heaters (and, on the output routes, a phase the detectors do not see) act on it."""
        if hasattr(self, "core"):
            return
        keys = {c["idx"]: c.get("key", "") for c in self.comps}
        conv = {}
        core = []
        for op in self.ops:
            k = keys[op[4]]
            if k.startswith("ch") and k.endswith("_enc_out"):
                conv[int(k[2:k.index("_")])] = op
            else:
                core.append(op)
        assert sorted(conv) == list(range(N))
        self.conv = [conv[i] for i in range(N)]
        self.core = core

    def transfer(self, phase):
        """(K x n_wg) phases -> (K x 8 x 3): field at each detector per unit field leaving each converger
        waveguide (route phases set by encoder heaters left out: they do not change detected power)."""
        self._split()
        K = phase.shape[0]
        f = np.zeros((self.n_nets, K * N), complex)
        for i, op in enumerate(self.conv):
            f[op[1][0], i::N] = 1.0
        e = np.exp(1j * np.repeat(phase, N, axis=0))
        for ins, outs, M, wgk, _ in self.core:
            v = M @ f[ins]
            if wgk >= 0:
                v = v * e[:, wgk]
            f[outs] = v
        return np.stack([f[n] * t for n, t in zip(self.det_net, self.t_out)], 1).reshape(K, N, 3)


def full_check(corner=F.NOMINAL, n=24, seed=0):
    """Forward model against the full SAX scattering solution (all reflections), random heater phases."""
    import jax.numpy as jnp
    import sax
    rng = np.random.default_rng(seed)
    circ = Circuit(corner)
    net = F.netlist(corner, EXTRACT)
    sc, _ = sax.circuit(netlist=net, models=F.MODELS)
    ph = rng.uniform(0, 2 * np.pi, (n, len(HEATERS)))
    phase = np.zeros((n, len(circ.wg)))
    for j, h in enumerate(HEATERS):
        phase[:, circ.wg_role[h]] = ph[:, j]
    det, _ = circ.run(phase)
    s = sc(wl=jnp.full(n, circ.wl), **{h + "_ps": {"phi": jnp.asarray(ph[:, j])} for j, h in enumerate(HEATERS)})
    full = np.stack([np.asarray(s["laser", d]) for d in DETS], 1)
    p1, p2 = np.abs(det) ** 2, np.abs(full) ** 2
    return dict(max_power=float(p2.max()), max_abs_power_difference=float(np.abs(p1 - p2).max()),
                max_relative_difference=float((np.abs(p1 - p2) / p2.max()).max()))


def crosstalk_matrix(circ, step=2.0):
    """X[w, h]: phase on waveguide component w per unit of heater h's own phase, from the drawn paths.
    A waveguide point at distance d from the heater (its nearest point on the heater line) adds
    XT0 exp(-(d - XT_D0) / XT_L) per unit length, normalised by the heater length, so a parallel
    waveguide of the heater's length at d picks up XT0 exp(-(d - XT_D0) / XT_L)."""
    comps = {c["idx"]: c for c in circ.comps}
    pts, owner, ds = [], [], []
    for k, ci in enumerate(circ.wg):
        path = np.asarray(comps[ci]["path"], float)
        for a, b in zip(path[:-1], path[1:]):
            L = float(np.hypot(*(b - a)))
            n = max(1, int(math.ceil(L / step)))
            t = (np.arange(n) + 0.5) / n
            pts.append(a[None] + t[:, None] * (b - a)[None])
            owner.append(np.full(n, k))
            ds.append(np.full(n, L / n))
    pts, owner, ds = np.concatenate(pts), np.concatenate(owner), np.concatenate(ds)
    X = np.zeros((len(circ.wg), len(HEATERS)))
    for h, name in enumerate(HEATERS):
        w = circ.wg_role[name]
        (x0, y0), (x1, y1) = comps[circ.wg[w]]["path"]
        Lh = math.hypot(x1 - x0, y1 - y0)
        tx = np.clip(((pts[:, 0] - x0) * (x1 - x0) + (pts[:, 1] - y0) * (y1 - y0)) / Lh ** 2, 0, 1)
        d = np.hypot(pts[:, 0] - (x0 + tx * (x1 - x0)), pts[:, 1] - (y0 + tx * (y1 - y0)))
        m = (d < XT_REACH) & (owner != w)
        g = XT0 * np.exp(-(d[m] - XT_D0) / XT_L) * ds[m] / Lh
        X[:, h] = np.bincount(owner[m], weights=g, minlength=len(circ.wg))
        X[w, h] = 1.0
    return X


class Die:
    """One simulated chip: corner circuit + heater efficiencies, static phases, crosstalk, noise."""

    def __init__(self, circ, X, rng):
        self.c, self.X, self.rng = circ, X, rng
        self.eta = (math.pi / P_PI) * (1 + ETA_SPREAD * rng.standard_normal(len(HEATERS)))
        self.phi0 = rng.uniform(0, 2 * np.pi, len(circ.wg))
        self.fs = self._full_scale()
        self.n_meas = 0

    def phases(self, P):
        return self.phi0[None] + (np.atleast_2d(P) * self.eta[None]) @ self.X.T

    def fields(self, P, chunk=4096):
        P = np.atleast_2d(P)
        out = [self.c.run(self.phases(P[a:a + chunk])) for a in range(0, len(P), chunk)]
        return np.concatenate([o[0] for o in out]), np.concatenate([o[1] for o in out])

    def power(self, P):
        return np.abs(self.fields(P)[0]) ** 2

    def page_power(self, Pe, Pm):
        """Detector powers (K x n x 3) for n encoder settings Pe (n x 16) under K mesh settings Pm (K x 56),
        the same as power() on every combination (encoder heaters do not reach the mesh section beyond
        the converger waveguides, mesh heaters do not reach the input section; checked in the tests)."""
        c = self.c
        c._split()
        ne = len(ENC_H)
        Pe, Pm = np.atleast_2d(Pe), np.atleast_2d(Pm)
        pe = (Pe * self.eta[None, :ne]) @ self.X[:, :ne].T                       # (n, n_wg) from encoders
        pm = (Pm * self.eta[None, ne:]) @ self.X[:, ne:].T                       # (K, n_wg) from the mesh
        enc = c._solve_inputs(self.phi0[None] + pe)                               # (n, 8): input section
        w = [op[3] for op in c.conv]
        t = np.array([op[2][0, 0] for op in c.conv])
        mid = enc[None] * t[None, None] * np.exp(1j * (self.phi0[w][None, None] + pe[:, w][None] + pm[:, w][:, None]))   # (K, n, 8)
        T = c.transfer(self.phi0[None] + pm)                                     # (K, 8, 3)
        return np.abs(np.einsum("kni,kid->knd", mid, T)) ** 2

    def page_measure(self, Pe, Pm):
        p = self.page_power(Pe, Pm)
        self.n_meas += p.shape[0] * p.shape[1]
        return p + MEAS_NOISE * self.fs[None, None] / N * self.rng.standard_normal(p.shape)

    def measure(self, P):
        p = self.power(P)
        self.n_meas += p.shape[0]
        return p + MEAS_NOISE * self.fs[None] / N * self.rng.standard_normal(p.shape)

    def _full_scale(self):
        """Detector reading if every encoder were fully open and all light reached that detector: total
        power into the mesh times the route's transmission. The reading noise is 0.2% of an eighth of
        this, the reading when one input's light all reaches one detector."""
        ph = self.phi0[None].copy()
        for i in range(N):                     # open each encoder: equal total phase on its two arms
            e = self.c.inp["encs"][i]
            sA, sB = ph[0, e["wgk"][0]].sum(), ph[0, e["wgk"][1]].sum()
            hB = self.c.wg_role[f"E{i}B"]
            ph[0, hB] += sA - sB
        det, out = self.c.run(ph)
        tot = float((np.abs(out) ** 2).sum())
        route = np.array([np.abs(det[0, k]) ** 2 / max(np.abs(out[0, ch]) ** 2, 1e-30) for k, ch in enumerate(DET_CH)])
        return tot * route


# ------------------------------------------------------------------ calibration: encoders
E_GRID = np.linspace(0, 70e-3, 17)


def _enc_model(par, PA, PB):
    etaA, etaB, kap, delta = par
    D = (1 - kap) * (etaA * PA - etaB * PB) + delta
    C = (1 + kap) * (etaA * PA + etaB * PB)
    c = np.cos(D / 2)
    return np.stack([np.ones_like(PA), c ** 2, c * np.cos(C / 2), c * np.sin(C / 2)], 1)


def _enc_resid(par, PA, PB, R):
    A = _enc_model(par, PA, PB)
    coef, *_ = np.linalg.lstsq(A, R, rcond=None)
    return (A @ coef - R).ravel()


def calibrate_encoder(die, j, base):
    """2-D sweep of encoder j's two heaters (others held at base), fitted with the push-pull model:
    arm phases eta_A P_A + kappa eta_B P_B and eta_B P_B + kappa eta_A P_A (kappa: heat each heater
    sends the other arm); the encoder's field ~ cos(D/2) exp(iC/2) with D the arms' phase difference."""
    from scipy.optimize import least_squares
    PA, PB = [g.ravel() for g in np.meshgrid(E_GRID, E_GRID, indexing="ij")]
    P = np.tile(base, (PA.size, 1))
    P[:, 2 * j], P[:, 2 * j + 1] = PA, PB
    R = die.measure(dac(P))
    eta0 = math.pi / P_PI
    best = None
    for d0 in np.linspace(0, 2 * np.pi, 6, endpoint=False):
        for k0 in (0.05, 0.15):
            try:
                r = least_squares(_enc_resid, [eta0, eta0, k0, d0], args=(PA, PB, R),
                                  bounds=([0.5 * eta0, 0.5 * eta0, -0.2, -10], [1.6 * eta0, 1.6 * eta0, 0.45, 20]))
            except Exception:
                continue
            if best is None or r.cost < best.cost:
                best = r
    etaA, etaB, kap, delta = best.x
    rms = float(np.sqrt(2 * best.cost / R.size) / die.fs.mean())
    return dict(etaA=float(etaA), etaB=float(etaB), kap=float(kap), delta=float(delta % (2 * np.pi)), fit_rms=rms)


def calibrate_encoders(die, rng):
    base = dac(rng.uniform(0, 50e-3, len(HEATERS)))
    enc = [calibrate_encoder(die, j, base) for j in range(N)]
    enc = refine_nulls(die, enc, rng)
    enc = calibrate_quadrature(die, enc, rng)
    enc[0]["mesh_periods"] = mesh_periods(die, enc, rng).tolist()
    return enc


def encoder_powers(enc, x, shift=None, dD=None):
    """Heater powers (B x 16) that set encoder j's output to x_j in [-1, 1] at a fixed output phase.
    shift (B x 8, optional) adds to both arms' phase sum, turning encoder j's output phase (used to
    measure the imaginary parts of the mesh's rows; enc[j]["quad"] is the shift for a quarter turn)."""
    x = np.clip(np.atleast_2d(x), -1, 1)
    P = np.zeros((x.shape[0], 2 * N))
    for j, e in enumerate(enc):
        D = 2 * np.arccos(x[:, j])                                  # 0 .. 2 pi
        if dD is not None:
            D = D + dD[:, j]
        u = (D - e["delta"]) / (1 - e["kap"])
        K = max(e["delta"], 2 * np.pi - e["delta"]) / (1 - e["kap"]) + 0.05
        if shift is not None:
            K = K + shift[:, j]
        P[:, 2 * j] = (K + u) / 2 / e["etaA"]
        P[:, 2 * j + 1] = (K - u) / 2 / e["etaB"]
    return dac(P)


def _with_mesh(Pe, Pm):
    return np.concatenate([Pe, np.broadcast_to(Pm, (Pe.shape[0], 56))], 1)


def refine_nulls(die, enc, rng, passes=2, settings=4):
    """Each encoder's zero, refined: with every other encoder at zero, sweep this one's phase difference
    over a full turn around its zero (four random mesh settings, all three detectors summed). The
    reading is |L + c sin((d - eps)/2)|^2, where L is what the other encoders still leak: the leak only
    adds terms at half the sweep frequency, so fitting both frequencies and taking eps from the full-
    frequency part finds the zero itself, not the setting that cancels the others' leak."""
    Pm = [dac(rng.uniform(0, 50e-3, 56)) for _ in range(settings)]
    dd = np.linspace(-np.pi, np.pi, 61)
    A = np.stack([np.ones_like(dd), np.cos(dd), np.sin(dd), np.cos(dd / 2), np.sin(dd / 2)], 1)
    for _ in range(passes):
        for k in range(N):
            e = enc[k]
            dD = np.zeros((dd.size, N))
            dD[:, k] = -dd
            rows = encoder_powers(enc, np.zeros((dd.size, N)), dD=dD)
            R = sum(die.measure(_with_mesh(rows, pm)).sum(1) for pm in Pm)
            c = np.linalg.lstsq(A, R, rcond=None)[0]
            eps = float(np.arctan2(-c[2], -c[1]))          # minimum of c1 cos d + c2 sin d
            e["delta"] = float((e["delta"] + eps) % (2 * np.pi))
            e["null_shift"] = e.get("null_shift", 0.0) + eps
    return enc


def calibrate_quadrature(die, enc, rng, settings=2):
    """For each encoder, the phase-sum shift that turns its output by a quarter period (to measure the
    imaginary parts of the mesh's rows). Encoder j and a partner both at +1, j's phase sum swept over a
    full turn: each detector sees a + b cos(g s) + c sin(g s); g, the output phase per unit of shift, is
    shared by the three detectors and two mesh settings and found by a grid search, then refined."""
    from scipy.optimize import least_squares
    s = np.linspace(0.0, 10.0, 51)
    Pm = [dac(rng.uniform(0, 50e-3, 56)) for _ in range(settings)]
    for j in range(N):
        l = (j + 1) % N
        x = np.zeros((s.size, N))
        x[:, [j, l]] = 1.0
        sh = np.zeros((s.size, N))
        sh[:, j] = s
        Pe = encoder_powers(enc, x, sh)
        R = np.concatenate([die.measure(_with_mesh(Pe, pm)) for pm in Pm], 1)          # (51, 3 * settings)

        def res(g):
            A = np.stack([np.ones_like(s), np.cos(g * s), np.sin(g * s)], 1)
            coef = np.linalg.lstsq(A, R, rcond=None)[0]
            return (A @ coef - R).ravel()
        grid = np.linspace(0.2, 1.2, 201)
        g0 = grid[int(np.argmin([np.sum(res(g) ** 2) for g in grid]))]
        g = float(least_squares(lambda v: res(v[0]), [g0], bounds=([0.15], [1.3])).x[0])
        enc[j]["rate"] = g
        enc[j]["quad"] = float(np.pi / 2 / g)
    return enc


def mesh_periods(die, enc, rng, probes=4, bases=3):
    """Each mesh heater's period (W per 2 pi of the response it drives), from sweeps over the driver's
    range with every other heater fixed (three random settings, four random inputs, three detectors):
    the detectors follow a sinusoid in that heater's power, at one rate shared by all the sweeps."""
    x = rng.uniform(-1, 1, (probes, N))
    Pe = encoder_powers(enc, x)
    base = [dac(rng.uniform(0, 50e-3, 56)) for _ in range(bases)]
    sweep = dac(np.linspace(0, DAC_MAX, 81))
    per = np.zeros(56)
    for k in range(56):
        Rs = []
        for b0 in base:
            Pm = np.tile(b0, (sweep.size, 1))
            Pm[:, k] = sweep
            Rs.append(die.page_measure(Pe, Pm).reshape(sweep.size, -1))
        R = np.concatenate(Rs, 1)                                        # (81, bases x probes x 3)
        best = None
        for g in np.linspace(2 * np.pi / 80e-3, 2 * np.pi / 30e-3, 251):     # shared rate, grid search
            A = np.stack([np.ones_like(sweep), np.cos(g * sweep), np.sin(g * sweep)], 1)
            r = np.sum((A @ np.linalg.lstsq(A, R, rcond=None)[0] - R) ** 2)
            if best is None or r < best[0]:
                best = (r, g)
        per[k] = 2 * np.pi / best[1]
        if not 31e-3 < per[k] < 79e-3:              # no clean response (e.g. heaters after the detected outputs)
            per[k] = 2 * P_PI
    return per


# ------------------------------------------------------------------ calibration: programming a page
PAIRS = [(j, l) for j in range(N) for l in range(j + 1, N)]
X36 = np.zeros((N + len(PAIRS), N))
X36[np.arange(N), np.arange(N)] = 1.0
for k, (j, l) in enumerate(PAIRS):
    X36[N + k, [j, l]] = 1.0
IU = np.triu_indices(N)
IU1 = np.triu_indices(N, 1)


def q_from(R, Rq=None):
    """Readings (..., 36, 3) for the 36 test inputs -> H matrices (..., 3, 8, 8): H_jl = conj(m_j) m_l.
    Without the quadrature readings Rq (..., 28, 3: input l turned a quarter period), only Re H."""
    single = R[..., :N, :]
    H = np.zeros(R.shape[:-2] + (R.shape[-1], N, N), complex)
    for d in range(R.shape[-1]):
        H[..., d, np.arange(N), np.arange(N)] = single[..., d]
        for k, (j, l) in enumerate(PAIRS):
            re = (R[..., N + k, d] - single[..., j, d] - single[..., l, d]) / 2
            im = 0.0 if Rq is None else -(Rq[..., k, d] - single[..., j, d] - single[..., l, d]) / 2
            H[..., d, j, l] = re + 1j * im
            H[..., d, l, j] = re - 1j * im
    return H


def q_target(rows):
    return np.array([np.conj(t)[:, None] * t[None, :] for t in rows])


def _norm(Q):
    tr = np.real(np.trace(Q, axis1=-2, axis2=-1))
    return Q / np.where(np.abs(tr) > 1e-30, tr, 1.0)[..., None, None]


class Page:
    """Measured-feedback programming of the mesh for up to three wanted rows (one per detector).

    Real inputs only ever see Re H (H_jl = conj(m_j) m_l). Fitting Re H alone has false minima: a row
    and its complex conjugate give the same Re H, and with three rows that must also be orthogonal a fit
    can settle on a mixture that fits none. So the fit runs in two stages. First the full complex rows
    (Re and Im H, the imaginary parts from inputs turned a quarter period by the encoders), added one
    row at a time. Then Re H alone, starting there, so the small error of the quarter turn does not stay
    in the result. A run whose measured misfit stays above the noise is shaken and refitted, and after
    several of those restarted from new random heater settings."""

    def __init__(self, die, enc, cscale, rows):
        self.die, self.rows = die, [np.asarray(r, complex) for r in rows]
        self.nd = len(rows)
        self.na = self.nd                                               # rows currently matched
        self.Pe = encoder_powers(enc, X36 * cscale[None])               # (36, 16)
        xq = X36[N:] * cscale[None]
        sh = np.zeros_like(xq)
        for k, (j, l) in enumerate(PAIRS):
            sh[k, l] = enc[l]["quad"]
        self.Peq = encoder_powers(enc, xq, sh)                          # (28, 16)
        self.Ht = _norm(q_target(self.rows))
        self.complex = False
        self.n_meas = 0
        self.period = np.asarray(enc[0].get("mesh_periods", np.full(56, 2 * P_PI)))

    def readings(self, Pm, truth=False, quad=None):
        quad = self.complex if quad is None else quad
        Pe = np.concatenate([self.Pe, self.Peq]) if quad else self.Pe
        Pm = dac(np.atleast_2d(Pm))
        self.n_meas += 0 if truth else Pm.shape[0] * len(Pe)
        return self.die.page_power(Pe, Pm) if truth else self.die.page_measure(Pe, Pm)

    def H(self, R):
        if R.shape[1] > len(X36):
            return _norm(q_from(R[:, :len(X36)], R[:, len(X36):])[:, :self.nd])
        return _norm(q_from(R)[:, :self.nd])

    def resid(self, R):
        D = (self.H(R) - self.Ht[None])[:, :self.na]
        parts = [D.real[:, :, IU[0], IU[1]]]
        if R.shape[1] > len(X36):
            parts.append(D.imag[:, :, IU1[0], IU1[1]])
        return np.concatenate([q.reshape(R.shape[0], -1) for q in parts], 1)

    def _jac(self, P, step):
        dP = np.where(P + step <= DAC_MAX, step, -step)
        rr = self.resid(self.readings(np.concatenate([P[None], P[None] + np.diag(dP)])))
        return rr[0], (rr[1:] - rr[0][None]).T / dP[None]

    def _lm(self, P, iters, step):
        """Levenberg-Marquardt on measured residuals; the Jacobian is measured each iteration (every
        heater nudged in turn: 57 settings x 36 inputs)."""
        n_it, lam, c = 0, 1e-2, np.inf
        while n_it < iters:
            n_it += 1
            r, J = self._jac(P, step)
            c = r @ r
            A, g = J.T @ J, J.T @ r
            improved = False
            damp = np.diag(np.diag(A) + 0.02 * np.trace(A) / len(A))      # floor: heaters the rows barely feel
            for _ in range(8):
                d = -np.linalg.solve(A + lam * damp, g)
                d *= min(1.0, MAX_STEP / max(np.abs(d).max(), 1e-30))       # trust region: at most MAX_STEP per heater
                Pn = P + d
                Pn = np.where(Pn > DAC_MAX, Pn - self.period, Pn)            # one period back: the same phase
                Pn = np.where(Pn < 0, Pn + self.period, Pn)
                Pn = dac(np.clip(Pn, 0, DAC_MAX))
                rn = self.resid(self.readings(Pn))[0]
                if rn @ rn < c:
                    P, c, improved = Pn, rn @ rn, True
                    lam = max(lam / 3, 1e-7)
                    break
                lam *= 4
            if not improved or c < 0.05 * ACCEPT * self.na:
                break
        return P, float(c), n_it

    def _fit(self, P, iters, step):
        self.complex = True
        for na in range(1, self.nd + 1):
            self.na = na
            P, c, n = self._lm(P, iters, step)
        self.complex = False
        return self._lm(P, iters, step)[:2]

    def run(self, rng, iters=40, step=2e-3, accept=None, starts=6, hops=3):
        accept = ACCEPT * self.nd if accept is None else accept
        best, tries, log = None, 0, []
        for attempt in range(starts):
            P, c = self._fit(dac(rng.uniform(0, 1, 56) * self.period), iters, step)
            tries += 1
            log.append(round(c, 5))
            for h in range(hops):
                if c < accept:
                    break
                Ph = P + rng.uniform(-0.25, 0.25, 56) * self.period
                Ph = np.where(Ph > DAC_MAX, Ph - self.period, np.where(Ph < 0, Ph + self.period, Ph))
                Ph, ch = self._fit(dac(Ph), iters, step)
                tries += 1
                log.append(round(ch, 5))
                if ch < c:
                    P, c = Ph, ch
            if best is None or c < best[1]:
                best = (P, c)
            if c < accept:
                break
        self.na, self.complex = self.nd, False
        return best[0], dict(cost=best[1], tries=tries, costs=log, accepted=bool(best[1] < accept), measurements=self.n_meas)

    def true_error(self, P):
        """Worst row: |Re H_true - Re H_wanted|_F / |Re H_wanted|_F, noiseless (what real inputs see)."""
        Q = self.H(self.readings(P, truth=True, quad=False))[0].real
        Qt = self.Ht.real
        return float(max(np.linalg.norm(Q[d] - Qt[d]) / np.linalg.norm(Qt[d]) for d in range(self.nd)))


# ------------------------------------------------------------------ the network on a die
def program_network(die, enc, net, rng):
    """Program all seven pages (3 + 3 + 1 rows of layer 1, 3 + 3 + 3 + 1 of layer 2). Returns per page:
    the mesh heater powers, each detector's scale (the measured trace: the reading per unit |t . x|^2)
    and the programming record."""
    pages = []
    for rows in list(net["pages1"]) + list(net["pages2"]):
        pg = Page(die, enc, np.ones(N), rows)
        P, info = pg.run(rng)
        R = pg.readings(np.repeat(P[None], 4, 0)).mean(0)          # final check, four readings averaged
        scale = R[:N].sum(0)
        info["true_row_error"] = pg.true_error(P)
        pages.append(dict(P=P, scale=scale, n=len(rows), info=info))
    return pages


def chip_layer(die, enc, pages, truth=False):
    """layer(k, rows, inputs) for mesh_net.forward: drive the encoders with the inputs, set the mesh to
    page k, read the detectors (with noise unless truth) and scale them."""
    def layer(k, rows, inp):
        pg = pages[k]
        P = _with_mesh(encoder_powers(enc, inp), pg["P"])
        R = die.power(P) if truth else die.measure(P)
        return (R / pg["scale"][None])[:, :pg["n"]]
    return layer


def run_die(corner, seed, net, A, y, ref, circ=None, X=None, detail=False):
    """One die: calibrate, program, classify the test digits. ref: the software network's answers."""
    import mesh_net
    t0 = time.time()
    circ = circ or Circuit(corner)
    X = crosstalk_matrix(circ) if X is None else X
    rng = np.random.default_rng(seed)
    die = Die(circ, X, rng)
    enc = calibrate_encoders(die, rng)
    n_enc = die.n_meas
    pages = program_network(die, enc, net, rng)
    logits, h, z, s_ = mesh_net.forward(net, A, chip_layer(die, enc, pages))
    pred = np.argmax(logits, 1)
    h_ref = mesh_net.forward(net, A)[1]
    r = dict(corner=list(corner), seed=seed, accuracy=float(np.mean(pred == y)), agreement=float(np.mean(pred == ref)),
             hidden_rms_error=float(np.sqrt(np.mean((h - h_ref) ** 2)) / np.sqrt(np.mean(h_ref ** 2))),
             page_row_errors=[round(p["info"]["true_row_error"], 4) for p in pages],
             page_tries=[p["info"]["tries"] for p in pages], pages_accepted=[p["info"]["accepted"] for p in pages],
             measurements_calibration=int(n_enc), measurements_programming=int(die.n_meas - n_enc - len(A) * len(pages)),
             encoder_null_shift=[round(e.get("null_shift", 0.0), 4) for e in enc], seconds=round(time.time() - t0, 1))
    if detail:
        r["detail"] = dict(h_chip=h[:400].round(5).tolist(), h_ref=h_ref[:400].round(5).tolist(),
                           logits_chip=logits[:400].round(4).tolist(), pred_chip=pred.tolist(),
                           encoders=enc, page_costs=[p["info"]["costs"] for p in pages],
                           page_P_mW=[(p["P"] * 1e3).round(4).tolist() for p in pages], page_scale=[p["scale"].tolist() for p in pages],
                           full_scale=die.fs.tolist(), eta=die.eta.tolist())
    return r


def _work(args):
    import mesh_net
    corner, seed, detail = args
    net, fm, _ = mesh_net.load()
    _, _, Xte, yte = mesh_net.mnist()
    A = mesh_net.features(fm, Xte)
    ref = np.argmax(mesh_net.forward(net, A)[0], 1)
    return run_die(tuple(corner), seed, net, A, yte, ref, detail=detail)


N_DIE = int(os.environ.get("MESH_DIES", "10"))


def main():
    import multiprocessing as mp
    import mesh_net
    t0 = time.time()
    parts = os.environ.get("MESH_PARTS", os.path.join(HERE, ".mesh_parts.jsonl"))
    done = {}
    if os.path.exists(parts):
        for line in open(parts):
            r = json.loads(line)
            done[(tuple(r["corner"]), r["seed"])] = r
    net, fm, info = mesh_net.load()
    jobs = [(c, 1000 * k + d, False) for k, c in enumerate(F.CORNERS) for d in range(N_DIE)]
    jobs.append((F.NOMINAL, 7, True))                       # the example die, with its details
    todo = [j for j in jobs if (tuple(j[0]), j[1]) not in done]
    print(f"{len(jobs) - len(todo)} dies done, {len(todo)} to go", flush=True)
    with mp.get_context("spawn").Pool(int(os.environ.get("MESH_PROCS", "2"))) as pool:
        for r in pool.imap_unordered(_work, todo):
            done[(tuple(r["corner"]), r["seed"])] = r
            with open(parts, "a") as f:
                f.write(json.dumps(r) + "\n")
            print(r["corner"], r["seed"], r["accuracy"], r["agreement"], round(r["hidden_rms_error"], 4), r["page_tries"],
                  f"{r['seconds']}s, total {time.time() - t0:.0f}s", flush=True)
    rows = [done[(tuple(c), s)] for c, s, d in jobs if not d]
    example = done[(tuple(F.NOMINAL), 7)]
    corners = []
    for c in F.CORNERS:
        rs = [r for r in rows if tuple(r["corner"]) == tuple(c)]
        acc = np.array([r["accuracy"] for r in rs])
        agr = np.array([r["agreement"] for r in rs])
        corners.append(dict(corner=list(c), dies=len(rs), accuracy_mean=float(acc.mean()), accuracy_min=float(acc.min()),
                            accuracy_max=float(acc.max()), agreement_mean=float(agr.mean()), agreement_min=float(agr.min()),
                            hidden_rms_error_mean=float(np.mean([r["hidden_rms_error"] for r in rs])),
                            row_error_median=float(np.median([e for r in rs for e in r["page_row_errors"]])),
                            row_error_worst=float(np.max([e for r in rs for e in r["page_row_errors"]])),
                            pages_not_accepted=int(sum(not a for r in rs for a in r["pages_accepted"])),
                            fits_per_page_mean=float(np.mean([t for r in rs for t in r["page_tries"]])),
                            measurements_per_die=float(np.mean([r["measurements_calibration"] + r["measurements_programming"] for r in rs]))))
    # reading noise vs laser power at heater speed: 0.2% of one input's full reading on 2 pA/rtHz at 1 MHz
    fs8 = float(np.mean(example["detail"]["full_scale"])) / N
    p_need = math.sqrt(2 * (F.DET["tia_pA"] * 1e-12) ** 2 * F.DET["bw"]) / (MEAS_NOISE * F.DET["resp"] * fs8)
    res = {"method": __doc__.strip().splitlines()[0], "netlist": os.path.relpath(EXTRACT, HERE),
           "die_model": dict(P_pi_mW=P_PI * 1e3, efficiency_spread=ETA_SPREAD, static_phase="uniform 0..2 pi on every waveguide",
                             crosstalk=f"{XT0:.0%} at {XT_D0:.0f} um, decay length {XT_L:.0f} um, integrated along the drawn paths",
                             dac=f"16-bit, 0-{DAC_MAX * 1e3:.0f} mW", measurement_noise=MEAS_NOISE,
                             noise_reference="reading when one input's light all reaches one detector",
                             one_input_full_reading_of_laser=fs8, laser_mW_for_this_noise_at_1MHz=p_need * 1e3),
           "model_check": dict(nominal=full_check(F.NOMINAL), corner_210_480=full_check((210.0, 480.0))),
           "network": dict(software_accuracy=info["accuracy"]["software_test"], linear_7_features=info["accuracy"]["linear_test"],
                           electronic_mlp_7_hidden=info["accuracy"]["mlp_test"], test_digits=10000),
           "corners": corners, "dies": rows, "example": example,
           "summary": dict(dies=len(rows), accuracy_mean=float(np.mean([r["accuracy"] for r in rows])),
                           accuracy_min=float(np.min([r["accuracy"] for r in rows])),
                           agreement_mean=float(np.mean([r["agreement"] for r in rows])),
                           agreement_min=float(np.min([r["agreement"] for r in rows])),
                           pages_not_accepted=int(sum(c["pages_not_accepted"] for c in corners)))}
    with open(os.path.join(HERE, "mesh_results.json"), "w") as f:
        json.dump(res, f, default=float)
    print(json.dumps(res["summary"]), json.dumps(res["model_check"]), f"{time.time() - t0:.0f}s")


if __name__ == "__main__" and len(sys.argv) == 1:
    main()


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "check":
    t = time.time()
    c = Circuit()
    print("compiled", len(c.ops), "ops,", c.n_nets, "nets in %.1f s" % (time.time() - t))
    t = time.time()
    print(full_check(), "%.1f s" % (time.time() - t))
