#!/usr/bin/env python3
"""
Time-domain model of a circuit from the SiEPIC extraction of the layout.

The circuit is the same one full_adder_sim.py solves in the frequency domain, re-expressed as a
delay-line scattering network (slowly varying envelope at 1550 nm):
  * Y-branches, broadband couplers, grating couplers, terminators: their PDK S-matrices at 1550 nm,
    applied instantly (their own group delay, about 1 ps each, is left out; the total latency is
    taken from the frequency-domain model instead, see group_delay()).
  * every drawn waveguide: a delay line with its drawn length, the PDK waveguide's group delay
    (n_g = 4.19) and its complex transmission at 1550 nm, so the carrier phase is exactly the one
    the frequency-domain model uses;
  * every heater section: the same, plus a phase that can change in time. Light crossing it at
    time t picks up exp(i phi(t)); in the Michelson arms it is crossed twice, on the way out and
    on the way back from the loop mirror, at two different times.
Back-reflections are kept: every port scatters into every port every time step.
"""
import json
import math
import os
import sys

import numpy as np
import scipy.sparse as sp

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
import jax.numpy as jnp  # noqa: E402
import full_adder_sim as F  # noqa: E402

C0 = 299792458.0
LAM_UM = 1.55


def _wg(length_um, corner=F.NOMINAL):
    """Complex transmission at 1550 nm and group delay (s) of a strip waveguide, from the PDK model."""
    d = 1e-4
    wl = jnp.array([LAM_UM - d, LAM_UM, LAM_UM + d])
    s = F.siepic.waveguide(wl=wl, length=length_um, loss=F.LOSS, width=corner[1], height=corner[0])
    t = np.asarray(s[("o0", "o1")])
    ph = np.unwrap(np.angle(t))
    w = 2 * np.pi * C0 / (np.asarray(wl) * 1e-6)
    return complex(t[1]), abs((ph[2] - ph[0]) / (w[2] - w[0]))


def _sdict(model, corner):
    th, w = corner
    kw = {"yb": dict(thickness=th, width=w), "bdc": dict(thickness=th, width=w), "gc": dict(thickness=th)}.get(model, {})
    if model == "term":
        return {("o0", "o0"): 0j}
    f = {"yb": F.siepic.y_branch, "bdc": F.siepic.bidirectional_coupler, "gc": F.siepic.grating_coupler}[model]
    s = f(wl=jnp.array([LAM_UM]), **kw)
    return {k: complex(np.asarray(v)[0]) for k, v in s.items()}


class TDCircuit:
    def __init__(self, extract=F.EXTRACT, bit=None, dt=0.02e-12, corner=F.NOMINAL, keep_roles=None):
        comps = [c for c in json.load(open(extract))["components"] if bit is None or c.get("bit") == bit]
        self.dt = dt
        # ---- real components and their ports
        ports, port_of, blocks = [], {}, []
        for c in comps:
            if c["kind"] in ("Waveguide", "wg_heater"):
                continue
            model, pm = F.PORTMAP[c["kind"]]
            names = sorted(set(pm.values()) | ({"o0"} if model == "gc" else set()))
            base = len(ports)
            for n in names:
                port_of[(c["idx"], n)] = len(ports)
                ports.append((c["idx"], n))
            s = _sdict(model, corner)
            blocks.append([(port_of[(c["idx"], a)], port_of[(c["idx"], b)], v) for (a, b), v in s.items()])
            if model == "gc":
                c["_fibre"] = port_of[(c["idx"], "o0")]
        P = len(ports)
        rows, cols, vals = [], [], []
        for blk in blocks:
            for a, b, v in blk:            # out[a] = S[a,b] in[b]   (sax: (in, out) keys are symmetric/reciprocal)
                if v != 0:
                    rows.append(b); cols.append(a); vals.append(v)
        self.S = sp.csr_matrix((vals, (rows, cols)), shape=(P, P), dtype=complex)
        self.P = P
        self.fibre = {c.get("role"): c["_fibre"] for c in comps if "_fibre" in c}
        # ---- nets: which pins touch
        netpins = {}
        for c in comps:
            if c["kind"] in ("Waveguide", "wg_heater"):
                for pin, net in c["pins"].items():
                    netpins.setdefault(net, []).append(("2p", c, pin))
            else:
                model, pm = F.PORTMAP[c["kind"]]
                for pin, net in c["pins"].items():
                    netpins.setdefault(net, []).append(("rp", port_of[(c["idx"], pm[pin])], pin))
        wgcache = {}

        def wgt(L):
            if L not in wgcache:
                wgcache[L] = _wg(L, corner)
            return wgcache[L]

        # ---- links: from each real port, walk through chains of waveguides / heaters to the next real port
        links = []
        for net, pins in netpins.items():
            for kind, obj, pin in pins:
                if kind != "rp":
                    continue
                p = obj
                F_, D, heat = 1 + 0j, 0.0, []
                cur_net = net
                prev = ("rp", p)
                while True:
                    other = [q for q in netpins[cur_net] if not (q[0] == prev[0] and (q[1] is prev[1] if q[0] == "2p" else q[1] == prev[1]))]
                    assert len(other) == 1, (cur_net, netpins[cur_net])
                    k2, o2, pin2 = other[0]
                    if k2 == "rp":
                        q = o2
                        break
                    t0, tg = wgt(o2["length_um"])
                    if o2["kind"] == "wg_heater":
                        heat.append([o2["role"], D + tg / 2])      # heater centre, delay from the start
                    F_ *= t0
                    D += tg
                    pin_out = "opt2" if pin2 == "opt1" else "opt1"
                    cur_net = o2["pins"][pin_out]
                    prev = ("2p", o2)
                links.append(dict(p=p, q=q, F=F_, D=D, heat=[(r, D - d) for r, d in heat]))
        self.links = links
        self.Dsteps = np.array([max(1, int(round(l["D"] / dt))) for l in links])
        self.R = int(self.Dsteps.max()) + 2
        self.lp = np.array([l["p"] for l in links])
        self.lq = np.array([l["q"] for l in links])
        self.lF = np.array([l["F"] for l in links])
        self.heat_terms = [(i, r, max(0, int(round(d / dt)))) for i, l in enumerate(links) for r, d in l["heat"]]
        self.roles = sorted({r for _, r, _ in self.heat_terms})

    def run(self, n_steps, phases, src_port, src, record):
        """phases: role -> array (n_steps,) or float (per-pass phase, rad). src: complex array (n_steps,)
        injected into fibre port src_port. Returns {name: complex field leaving that fibre port}."""
        R = self.R
        hist = np.zeros((R, self.P), complex)
        ph = {r: (np.full(n_steps, float(v)) if np.isscalar(v) else np.asarray(v, float)) for r, v in phases.items()}
        for r in self.roles:
            ph.setdefault(r, np.zeros(n_steps))
        terms = {}
        for i, r, d in self.heat_terms:
            terms.setdefault(i, []).append((r, d))
        tl = np.array(sorted(terms))
        out = {k: np.zeros(n_steps, complex) for k in record}
        rec = {k: self.fibre[k] for k in record}
        Fbase = self.lF.copy()
        for n in range(n_steps):
            fac = Fbase.copy()
            for i in tl:
                s = 0.0
                for r, d in terms[i]:
                    s += ph[r][max(0, n - d)]
                fac[i] = fac[i] * np.exp(1j * s)
            vin = np.zeros(self.P, complex)
            np.add.at(vin, self.lq, fac * hist[(n - self.Dsteps) % R, self.lp])
            vin[src_port] += src[n]
            vout = self.S @ vin
            hist[n % R] = vout
            for k, pi in rec.items():
                out[k][n] = vout[pi]
        return out
