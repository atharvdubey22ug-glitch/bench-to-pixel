#!/usr/bin/env python3
"""Frequency-domain model and calibration of the single-pass XOR (klayout/fast_xor_extracted_netlist.json)."""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
import jax.numpy as jnp  # noqa: E402
import sax  # noqa: E402
import full_adder_sim as F  # noqa: E402

EXTRACT = os.path.join(HERE, "..", "klayout", "fast_xor_extracted_netlist.json")
ROLES = ["A_mod", "B_mod", "biasA", "biasB"]


class FastXOR:
    def __init__(self, corner=F.NOMINAL):
        self.circ, _ = sax.circuit(netlist=F.netlist(corner, EXTRACT), models=F.MODELS)

    def run(self, **ph):
        arrs = np.broadcast_arrays(*[np.atleast_1d(np.asarray(ph.get(k, 0.0), float)) for k in ROLES])
        shp = arrs[0].shape
        kw = {k + "_ps": {"phi": jnp.asarray(a.ravel())} for k, a in zip(ROLES, arrs)}
        s = self.circ(wl=jnp.full(arrs[0].size, 1.55), **kw)
        return {o: (np.abs(np.asarray(s["laser1", {"xor": "xor1"}.get(o, o)])) ** 2).reshape(shp) for o in ("xor", "mon")}

    def calibrate(self):
        """Grating-coupler measurements only. Total output (xor + mon) depends only on the light let
        through the two switches, so each switch is found by a sweep of its own heater; then the bias
        is set for a dark XOR output with both arms on."""
        th = np.linspace(0, 2 * np.pi, 721, endpoint=False)
        cal = {"biasB": 0.0}
        tot = lambda r: r["xor"] + r["mon"]  # noqa: E731
        p = tot(self.run(A_mod=th, B_mod=0.0))
        cal["A_mod_off"], cal["A_mod_on"] = float(th[np.argmin(p)]), float(th[np.argmax(p)])
        p = tot(self.run(A_mod=cal["A_mod_off"], B_mod=th))
        cal["B_mod_off"], cal["B_mod_on"] = float(th[np.argmin(p)]), float(th[np.argmax(p)])
        p = self.run(A_mod=cal["A_mod_on"], B_mod=cal["B_mod_on"], biasA=th)["xor"]
        b0 = float(th[np.argmin(p)])
        fine = b0 + np.linspace(-0.01, 0.01, 201)
        cal["biasA"] = float(fine[np.argmin(self.run(A_mod=cal["A_mod_on"], B_mod=cal["B_mod_on"], biasA=fine)["xor"])])
        self.cal = cal
        return cal

    def truth(self):
        c = self.cal
        out = {}
        for a in (0, 1):
            for b in (0, 1):
                r = self.run(A_mod=c["A_mod_on" if a else "A_mod_off"], B_mod=c["B_mod_on" if b else "B_mod_off"],
                             biasA=c["biasA"], biasB=c["biasB"])
                out[f"{a}{b}"] = {k: float(v[0]) for k, v in r.items()}
        return out
