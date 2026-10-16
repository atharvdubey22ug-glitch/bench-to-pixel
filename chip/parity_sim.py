#!/usr/bin/env python3
"""
Phase-encoded 8-bit parity chip (klayout/EBeam_heaters_atharvdubey_Parity8): circuit simulation.

The circuit is read from the SiEPIC extraction of the layout (klayout/parity_extracted_netlist.json)
and solved with the PDK S-matrices, exactly as for the full adder (full_adder_sim.netlist()).

What a real die adds on top of the drawing, and is modelled here:
  * heater efficiency: each TiW heater turns power into phase with its own efficiency,
    nominally pi per 25 mW, spread +-10% (1 sigma) from die to die and heater to heater;
  * a random static phase between the two arms (local width and thickness variation), uniform 0..2pi;
  * thermal crosstalk: each heater also warms its neighbours on the same arm (5% at 150 um, 1% at
    300 um) and the reference arm 146 um away (2%);
  * the process corner (silicon thickness 210/220/230 nm x waveguide width 480/500/520 nm), which
    changes every component, the couplers' split ratio included;
  * calibration from what can be measured: the power at the PARITY grating coupler, read with 0.2%
    noise, while one heater is swept at a time (16-bit DAC, 0..80 mW).

Results (parity_results.json):
  truth        every one of the 256 inputs on 20 random dies at each of the 9 corners, after calibration
               (light levels of every input for the first die of each corner, 1 mW laser; input index i
               has bit k = (i >> k) & 1)
  spectral     on/off ratio of the PARITY output from 1500 to 1600 nm
  accumulation how phase errors in the drive add up along the chain (the cost of having no regeneration),
               from 1 to 32 sections, and the same check through the full circuit model at 8 sections
  cascade      side by side with the amplitude-encoded cascade of the full adder (cascade_failure())
"""
import json
import math
import os
import sys
import warnings

import numpy as np
from scipy.optimize import curve_fit

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import jax.numpy as jnp  # noqa: E402
import sax  # noqa: E402
import full_adder_sim as F  # noqa: E402

EXTRACT = os.path.join(HERE, "klayout", "parity_extracted_netlist.json")
N = 8
SECT = [f"S{k + 1}" for k in range(N)]
ROLES = SECT + ["bias"]
P_PI = 25e-3                    # W per pi, nominal TiW heater
ETA_SPREAD = 0.10
XT_SAME = {1: 0.05, 2: 0.01}    # crosstalk onto sections 1 and 2 pitches away
XT_REF = 0.02
DAC_MAX, DAC_BITS = 80e-3, 16   # 0..80 mW: a full 2*pi of bias even on a weak heater
MEAS_NOISE = 2e-3


class Circuit:
    """The drawn circuit at one corner: phases (rad) per heater -> field at PARITY and NOT PARITY."""

    def __init__(self, corner=F.NOMINAL, extract=EXTRACT):
        self.circ, _ = sax.circuit(netlist=F.netlist(corner, extract), models=F.MODELS)

    def fields(self, phases, wl=1.55):
        arrs = np.broadcast_arrays(*[np.atleast_1d(np.asarray(phases.get(k, 0.0), float)) for k in ROLES])
        n = arrs[0].size
        wl = np.broadcast_to(np.asarray(wl, float), (n,)) if np.ndim(wl) == 0 else np.asarray(wl)
        s = self.circ(wl=jnp.asarray(wl), **{k + "_ps": {"phi": jnp.asarray(a.ravel())} for k, a in zip(ROLES, arrs)})
        return {o: np.asarray(s["laser1", o]).reshape(arrs[0].shape) for o in ("parity", "nparity")}

    def power(self, phases, **kw):
        return {k: np.abs(v) ** 2 for k, v in self.fields(phases, **kw).items()}


class Die:
    """One fabricated die: heater powers (W) -> phases, with spread, arm offset and crosstalk."""

    def __init__(self, circuit, rng):
        self.c = circuit
        self.eta = (math.pi / P_PI) * (1 + ETA_SPREAD * rng.standard_normal(len(ROLES)))
        self.phi0 = rng.uniform(0, 2 * np.pi)
        X = np.eye(len(ROLES))
        for i in range(len(ROLES)):
            for j in range(len(ROLES)):
                X[i, j] += XT_SAME.get(abs(i - j), 0.0)
        self.X = X
        self.rng = rng

    def phases(self, powers):
        """powers: (..., 9) array of heater powers -> dict of section phases. The section's phase is
        its own efficiency times the heat reaching it; heat reaching the reference arm is subtracted
        from the bias section (only the difference between the arms matters)."""
        pw = np.asarray(powers, float)
        heat = pw @ self.X.T
        ph = heat * self.eta
        ph[..., -1] += self.phi0 - XT_REF * (pw * self.eta).sum(-1)
        return {k: ph[..., i] for i, k in enumerate(ROLES)}

    def measure(self, powers):
        p = self.c.power(self.phases(powers))["parity"]
        return p * (1 + MEAS_NOISE * self.rng.standard_normal(p.shape))


def dac(x):
    q = DAC_MAX / (2 ** DAC_BITS - 1)
    return np.clip(np.round(np.asarray(x) / q) * q, 0, DAC_MAX)


def _fit(x, y):
    """y = a + b cos(g x + d): returns (g, d) with b > 0."""
    f = np.fft.rfft(y - y.mean())
    k = int(np.argmax(np.abs(f[1:]))) + 1
    g0 = 2 * np.pi * k / (x[-1] - x[0] + (x[1] - x[0]))
    best = None
    for gs in (g0 * 0.9, g0, g0 * 1.1):
        try:
            p, _ = curve_fit(lambda x, a, b, g, d: a + b * np.cos(g * x + d), x, y,
                             p0=[y.mean(), (y.max() - y.min()) / 2, gs, 0.0], maxfev=20000)
        except RuntimeError:
            continue
        r = np.sum((p[0] + p[1] * np.cos(p[2] * x + p[3]) - y) ** 2)
        if best is None or r < best[0]:
            best = (r, p)
    a, b, g, d = best[1]
    if g < 0:
        g, d = -g, -d
    if b < 0:
        d += np.pi
    return g, d % (2 * np.pi)


def calibrate(die):
    """1. all sections off, sweep the bias heater: set it where PARITY is darkest.
       2. each section alone: the smallest power that makes PARITY brightest is its '1' level."""
    x = dac(np.linspace(0, DAC_MAX, 601))
    pw = np.zeros((x.size, len(ROLES)))
    pw[:, -1] = x
    g, d = _fit(x, die.measure(pw))
    bias = dac(((np.pi - d) % (2 * np.pi)) / g)               # g x + d = pi: minimum
    on = []
    for k in range(N):
        pw = np.zeros((x.size, len(ROLES)))
        pw[:, -1] = bias
        pw[:, k] = x
        gk, dk = _fit(x, die.measure(pw))
        # bias puts the k = 0 point at the minimum, so the first maximum is half a period later
        on.append(dac(((0.0 - dk) % (2 * np.pi)) / gk))
    return dict(bias_mW=float(bias * 1e3), on_mW=[float(v * 1e3) for v in on])


def bits_table(n=N):
    return ((np.arange(2 ** n)[:, None] >> np.arange(n)) & 1).astype(int)


def evaluate(die, cal, laser_W=1e-3):
    B = bits_table()
    pw = np.zeros((B.shape[0], len(ROLES)))
    pw[:, :N] = B * (np.array(cal["on_mW"]) * 1e-3)
    pw[:, -1] = cal["bias_mW"] * 1e-3
    p = die.c.power(die.phases(pw))
    par = B.sum(1) % 2
    one, zero = p["parity"][par == 1], p["parity"][par == 0]
    diff = p["parity"] - p["nparity"]                            # balanced detection
    d1, d0 = diff[par == 1], diff[par == 0]
    thr = 0.5 * (one.min() + zero.max())
    return dict(levels_uW=dict(parity=(p["parity"] * laser_W * 1e6).round(5).tolist(), nparity=(p["nparity"] * laser_W * 1e6).round(5).tolist()),
                correct=bool(one.min() > zero.max() and d1.min() > 0 > d0.max()),
                on_off_dB=float(10 * np.log10(one.min() / max(zero.max(), 1e-12))),
                one_uW=[float(one.min() * laser_W * 1e6), float(one.max() * laser_W * 1e6)],
                zero_uW=[float(zero.min() * laser_W * 1e6), float(zero.max() * laser_W * 1e6)],
                margin_pct=float(100 * (one.min() - zero.max()) / one.mean()),
                balanced_margin_pct=float(100 * min(d1.min(), -d0.max()) / d1.mean()),
                threshold_uW=float(thr * laser_W * 1e6))


def truth(n_dies=20, seed=7):
    out = {}
    for corner in F.CORNERS:
        c = Circuit(corner)
        rng = np.random.default_rng(seed + int(corner[0]) * 1000 + int(corner[1]))
        res = []
        for _ in range(n_dies):
            die = Die(c, rng)
            cal = calibrate(die)
            ev = evaluate(die, cal)
            ev["phase_error_deg"] = residual_error(die, cal)
            ev["calibration"] = cal
            if res:
                ev.pop("levels_uW")             # kept for the first die of each corner only
            res.append(ev)
        out[f"{int(corner[0])}nm_x_{int(corner[1])}nm"] = dict(
            dies=n_dies, all_correct=all(r["correct"] for r in res),
            worst_on_off_dB=round(min(r["on_off_dB"] for r in res), 1),
            median_on_off_dB=round(float(np.median([r["on_off_dB"] for r in res])), 1),
            worst_margin_pct=round(min(r["margin_pct"] for r in res), 1),
            worst_balanced_margin_pct=round(min(r["balanced_margin_pct"] for r in res), 1),
            worst_section_phase_error_deg=round(max(r["phase_error_deg"] for r in res), 2),
            one_level_uW_per_mW=round(float(np.median([r["one_uW"][0] for r in res])), 1),
            example=res[0])
        print(corner, out[list(out)[-1]]["all_correct"], out[list(out)[-1]]["worst_on_off_dB"], out[list(out)[-1]]["worst_margin_pct"])
    return out


def residual_error(die, cal):
    """Largest error, after calibration, of any section's '1' phase (ideal: exactly pi) and of the bias."""
    base = np.zeros(len(ROLES))
    base[-1] = cal["bias_mW"] * 1e-3
    ph0 = die.phases(base)
    tot0 = sum(ph0[k] for k in ROLES)
    err = []
    for k in range(N):
        pw = base.copy()
        pw[k] = cal["on_mW"][k] * 1e-3
        ph = die.phases(pw)
        d = sum(ph[r] for r in ROLES) - tot0
        err.append(abs(((d - np.pi + np.pi) % (2 * np.pi)) - np.pi))
    return float(np.degrees(max(err)))


def spectral():
    c = Circuit()
    wl = np.linspace(1.50, 1.60, 1001)
    # balanced arms, ideal calibration: bias found at 1550 nm, '1' = pi on S1
    th = np.linspace(0, 2 * np.pi, 3601)
    b = th[np.argmin(c.power({"bias": th})["parity"])]
    p0 = c.power({"bias": np.full(wl.size, b)}, wl=wl)["parity"]
    p1 = c.power({"bias": np.full(wl.size, b), "S1": np.full(wl.size, np.pi)}, wl=wl)["parity"]
    r = 10 * np.log10(p1 / p0)
    return dict({f"+-{d}nm": round(float(r[np.abs(wl - 1.55) <= d * 1e-3 + 1e-9].min()), 1) for d in (1, 5, 10, 20, 50)},
                wavelength_nm=(wl * 1e3)[::10].round(1).tolist(), on_off_dB=r[::10].round(2).tolist())


def accumulation():
    """Drive errors: each section's '1' phase is pi + e_k. The output carries the sum of the errors of
    the sections that are on, so errors add along the chain (no regeneration). Worst input = the one
    that sums the largest error. Ideal interferometer (the drawn one is balanced: see spectral)."""
    rng = np.random.default_rng(3)
    out = {"sections": list(range(1, 33))}
    for name, sig in (("random_2pct", 0.02), ("random_5pct", 0.05), ("random_10pct", 0.10)):
        worst = []
        for n in out["sections"]:
            e = sig * np.pi * rng.standard_normal((4000, n))
            # worst case over inputs: all positive errors on (or all negative)
            tot = np.maximum(np.clip(e, 0, None).sum(1), -np.clip(e, None, 0).sum(1))
            leak = np.sin(tot / 2) ** 2                  # '0' level that should be dark
            sig1 = np.cos(tot / 2) ** 2                  # '1' level that should be bright
            er = 10 * np.log10(np.maximum(sig1, 1e-9) / np.maximum(leak, 1e-9))
            worst.append(round(float(np.percentile(er, 5)), 2))   # 95% of dies are at least this good
        out[name] = worst
    for name, d in (("systematic_2pct", 0.02), ("systematic_5pct", 0.05)):
        tot = d * np.pi * np.array(out["sections"])
        out[name] = (10 * np.log10(np.cos(tot / 2) ** 2 / np.maximum(np.sin(tot / 2) ** 2, 1e-9))).round(2).tolist()
    # the same through the full circuit model, 8 sections, every input, 5% random error on each section
    c = Circuit()
    th = np.linspace(0, 2 * np.pi, 3601)
    b = th[np.argmin(c.power({"bias": th})["parity"])]
    B = bits_table()
    errs = []
    for _ in range(50):
        e = 0.05 * np.pi * rng.standard_normal(N)
        ph = {k: B[:, i] * (np.pi + e[i]) for i, k in enumerate(SECT)}
        ph["bias"] = np.full(B.shape[0], b)
        p = c.power(ph)["parity"]
        par = B.sum(1) % 2
        errs.append(10 * np.log10(p[par == 1].min() / p[par == 0].max()))
        tot = np.maximum(np.clip(e, 0, None).sum(), -np.clip(e, None, 0).sum())
        errs[-1] = (errs[-1], 10 * np.log10(np.cos(tot / 2) ** 2 / np.sin(tot / 2) ** 2))
    out["circuit_check_8_sections_5pct"] = dict(circuit_dB=[round(a, 2) for a, _ in errs], ideal_model_dB=[round(b_, 2) for _, b_ in errs])
    return out


def main():
    res = {"method": __doc__.strip().splitlines()[0], "netlist": os.path.relpath(EXTRACT, HERE),
           "die_model": dict(P_pi_mW=P_PI * 1e3, efficiency_spread=ETA_SPREAD, crosstalk_same_arm=XT_SAME, crosstalk_reference_arm=XT_REF,
                             dac=f"{DAC_BITS}-bit, 0-{DAC_MAX * 1e3:.0f} mW", measurement_noise=MEAS_NOISE)}
    res["truth"] = truth()
    res["spectral"] = spectral()
    print("spectral", {k: v for k, v in res["spectral"].items() if k.startswith("+-")})
    res["accumulation"] = acc = accumulation()
    print("accumulation 8 sections:", {k: acc[k][7] for k in acc if isinstance(acc[k], list) and k != "sections"})
    cf = json.load(open(os.path.join(HERE, "full_adder_results.json")))["cascade_failure"]
    res["cascade"] = dict(amplitude_encoded=dict(stages_that_work=1, relative_phase_A_only_vs_B_only_deg=round(cf["relative_phase_deg"], 1),
                                                 best_worst_case_dark_level=round(cf["best_worst_case"], 2)),
                          phase_encoded=dict(stages_that_work=N, note="8 sections in one pass; see truth"))
    with open(os.path.join(HERE, "parity_results.json"), "w") as f:
        json.dump(res, f, indent=1, default=float)


if __name__ == "__main__":
    main()
