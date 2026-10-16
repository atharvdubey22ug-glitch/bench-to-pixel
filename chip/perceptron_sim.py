#!/usr/bin/env python3
"""
Optical binary neuron (klayout/EBeam_heaters_atharvdubey_Perceptron16): circuit simulation.

The chip computes S = sum_k w_k x_k over 16 arms with w, x in {-1, +1}: arm k's phase section is set
to 0 when x_k = w_k and to pi when they differ, the combiner tree adds the 16 fields, and the
difference of the two output photocurrents, D = P+ - P-, is proportional to S (balanced homodyne
detection against the reference arm). S = 16 - 2 * (number of arms whose input differs from its
weight), so neighbouring values of S are one "level" (2 units) apart.

The circuit is read from the SiEPIC extraction of the layout and solved with the PDK component
models (full_adder_sim.netlist()). For speed, the fields at the two outputs are also written as a
sum over the 17 arms, out = sum_j t_j exp(i phi_j); linear_check() measures how well that matches
the full scattering solution (back-reflections included) before it is used.

A simulated die adds:
  * heater efficiency pi per 25 mW, spread +-10% (1 sigma);
  * a random static phase on every arm and on the reference (uniform 0..2 pi);
  * thermal crosstalk from each heater onto every waveguide running past it, by distance:
    5% at the 100 um arm pitch, 0.9% at 200 um, 0.2% at 300 um (exp. decay, 60 um length);
  * process corners (silicon 210/220/230 nm x width 480/500/520 nm).
Calibration uses only the two output grating couplers (0.2% reading noise, 16-bit 0..80 mW heater
drivers), in three steps:
  1. phases: each arm's heater is swept in turn and set where the difference signal is largest (the
     arm in phase with the reference = +1); -1 is half a period away. Three passes, because each
     heater also warms its neighbours (calibrate()).
  2. crosstalk map: how much each heater turns its neighbours and the reference, measured with the
     neighbour a quarter period off (measure_crosstalk()); the driver then shifts every heater so
     the heat that changes with the data is cancelled, pattern by pattern (powers_for()).
  3. balance weight: P- is weighted so the readout is a straight line in S even when the output
     coupler is not exactly 50/50 (balance_weight()).
"Uncorrected" results use step 1 alone and the plain difference P+ - P-.

Results (perceptron_results.json):
  linearity    D against S on 20 dies per corner: 7,522 arm patterns (470 for each value of S) through the
               full model, with and without steps 2 and 3, and all 65,536 through the linear model
  digits       a 3x5-pixel digit font as stored weight patterns (one pass of light per digit); every
               digit with 0, 1 and 2 pixels flipped, the winner read from the chip's 10 outputs
  spectral     D for S = +16 and -16 from 1500 to 1600 nm; path_delay: signal minus reference group delay
  noise        detector noise and the laser power needed to resolve one level
"""
import json
import math
import os
import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import jax.numpy as jnp  # noqa: E402
import sax  # noqa: E402
import full_adder_sim as F  # noqa: E402
from parity_sim import _fit, dac, DAC_MAX  # noqa: E402

EXTRACT = os.path.join(HERE, "klayout", "perceptron_extracted_netlist.json")
N = 16
ARMS = [f"A{k:02d}" for k in range(N)]
ROLES = ARMS + ["REF"]
P_PI, ETA_SPREAD, MEAS_NOISE = 25e-3, 0.10, 2e-3
XT0, XT_D0, XT_L = 0.05, 100.0, 60.0

# 3x5 digit font, rows top to bottom; pixel i = 3 * row + col drives arm 15 - i; arm 0 is the bias input
FONT = {0: "111101101101111", 1: "010110010010111", 2: "111001111100111", 3: "111001111001111", 4: "101101111001001",
        5: "111100111001111", 6: "111100111101111", 7: "111001001001001", 8: "111101111101111", 9: "111101111001111"}
ARM_OF_PIXEL = [15 - i for i in range(15)]
BIAS_ARM = 0


def geometry():
    """Heater centres (um) and the y of every waveguide they warm (arms and the reference line)."""
    comps = json.load(open(EXTRACT))["components"]
    pos = {c["role"]: c["center"] for c in comps if c["kind"] == "wg_heater"}
    return pos


def crosstalk_matrix(pos):
    """X[j, k]: phase on waveguide j per unit of heater k's own phase. Heater k sits on waveguide k;
    every other waveguide runs past it at |dy|. The reference runs past the arm heaters too."""
    y = np.array([pos[r][1] for r in ROLES])
    X = np.eye(len(ROLES))
    for j in range(len(ROLES)):
        for k in range(len(ROLES)):
            if j == k:
                continue
            d = abs(y[j] - y[k])
            if ROLES[k] == "REF" or ROLES[j] == "REF":
                # the reference line runs past the arm heaters' x; the reference heater sits beside the
                # split tree, where no single arm runs: only its own line is counted
                d = abs(y[k] - y[ROLES.index("REF")]) if ROLES[j] == "REF" else 1e9
            X[j, k] = XT0 * math.exp(-(d - XT_D0) / XT_L) if d < 1e8 else 0.0
    return X


class Circuit:
    def __init__(self, corner=F.NOMINAL):
        self.corner = corner
        self.circ, _ = sax.circuit(netlist=F.netlist(corner, EXTRACT), models=F.MODELS)
        self.t = self._linear()

    def fields(self, phases, wl=1.55):
        arrs = np.broadcast_arrays(*[np.atleast_1d(np.asarray(phases.get(k, 0.0), float)) for k in ROLES])
        n = arrs[0].size
        w = np.full(n, wl) if np.ndim(wl) == 0 else np.asarray(wl, float)
        out = {"plus": np.empty(n, complex), "minus": np.empty(n, complex)}
        for a in range(0, n, 4000):
            s = self.circ(wl=jnp.asarray(w[a:a + 4000]), **{k + "_ps": {"phi": jnp.asarray(v.ravel()[a:a + 4000])} for k, v in zip(ROLES, arrs)})
            for o in out:
                out[o][a:a + 4000] = np.asarray(s["laser1", o])
        return {o: v.reshape(arrs[0].shape) for o, v in out.items()}

    def _linear(self):
        """t[o][j]: field at output o from waveguide j, with every phase at 0 (toggle each by pi)."""
        ph = {r: np.zeros(len(ROLES) + 1) for r in ROLES}
        for j, r in enumerate(ROLES):
            ph[r][j + 1] = np.pi
        f = self.fields(ph)
        return {o: (f[o][0] - f[o][1:]) / 2 for o in f}

    def lin_fields(self, phi):
        """phi: (..., 17) phases -> fields from the linear model."""
        e = np.exp(1j * np.asarray(phi))
        return {o: e @ self.t[o] for o in self.t}

    def full_fields(self, phi):
        """phi: (..., 17) phases -> fields from the full scattering solution."""
        phi = np.asarray(phi)
        f = self.fields({r: phi[..., j].ravel() for j, r in enumerate(ROLES)})
        return {o: v.reshape(phi.shape[:-1]) for o, v in f.items()}


def diff_of(f, lam=1.0):
    return np.abs(f["plus"]) ** 2 - lam * np.abs(f["minus"]) ** 2


def readout(f, cal):
    """The balanced output: P+ minus the calibrated balance weight times P-."""
    return diff_of(f, cal.get("lam", 1.0))


class Die:
    def __init__(self, circuit, rng, X):
        self.c, self.rng, self.X = circuit, rng, X
        self.eta = (math.pi / P_PI) * (1 + ETA_SPREAD * rng.standard_normal(len(ROLES)))
        self.phi0 = rng.uniform(0, 2 * np.pi, len(ROLES))

    def phases(self, powers):
        return (np.asarray(powers) * self.eta) @ self.X.T + self.phi0

    def fields(self, powers, full=True):
        ph = self.phases(powers)
        return self.c.full_fields(ph) if full else self.c.lin_fields(ph)

    def measure(self, powers):
        d = diff_of(self.fields(powers))
        return d + MEAS_NOISE * np.abs(d).max() * self.rng.standard_normal(d.shape)

    def measure_pm(self, powers):
        f = self.fields(powers)
        pp, pm = np.abs(f["plus"]) ** 2, np.abs(f["minus"]) ** 2
        sc = MEAS_NOISE * max(pp.max(), pm.max())
        return pp + sc * self.rng.standard_normal(pp.shape), pm + sc * self.rng.standard_normal(pm.shape)


def calibrate(die, passes=3):
    p = np.zeros(len(ROLES))
    x = dac(np.linspace(0, DAC_MAX, 401))
    g_all = np.zeros(N)
    for _ in range(passes):
        for j in range(N):
            pw = np.tile(p, (x.size, 1))
            pw[:, j] = x
            g, d = _fit(x, die.measure(pw))
            p[j] = float(dac(((-d) % (2 * np.pi)) / g))
            g_all[j] = g
    half = np.pi / g_all
    alt = np.where(p[:N] + half <= DAC_MAX, p[:N] + half, p[:N] - half)
    return dict(p_plus=p.copy(), p_minus=np.concatenate([dac(alt), p[N:]]), g=g_all)


def measure_crosstalk(die, cal, repeats=16, reach=2):
    """Crosstalk map from the two output readings only. M[j, k]: phase of arm j, relative to the
    reference, per mW on heater k (j != k). Arm j is set a quarter period off its aligned point, where
    the output responds linearly to its phase, and heater k is switched between its +1 and -1 powers;
    repeating with arm j a quarter period the other way and subtracting cancels arm k's own flip and
    every second-order term. Measured for the arms within `reach` of k, plus one far arm whose shift
    is the heat reaching the reference (applied to every arm beyond reach). Also returns B[j], arm j's
    swing in the difference signal."""
    g, pp = cal["g"], cal["p_plus"]
    dP = cal["p_minus"][:N] - pp[:N]
    cfg = []
    for j in range(N):
        a = pp.copy(); b = pp.copy(); b[j] = cal["p_minus"][j]
        cfg += [a, b]
    Dm = die.measure(np.repeat(np.array(cfg), repeats, axis=0)).reshape(N, 2, repeats).mean(-1)
    B = (Dm[:, 0] - Dm[:, 1]) / 2

    def quarter(j, sgn):
        v = pp[j] + sgn * (np.pi / 2) / g[j]
        if v < 0:
            v += 2 * np.pi / g[j]
        if v > DAC_MAX:
            v -= 2 * np.pi / g[j]
        return v

    pairs = []
    for k in range(N):
        far = N - 1 if k < N // 2 else 0
        pairs += [(j, k) for j in range(N) if 0 < abs(j - k) <= reach] + [(far, k)]
    cfg = []
    for j, k in pairs:
        for sj in (1, -1):
            for kk in (0, 1):
                p = pp.copy()
                p[j] = quarter(j, sj)
                if kk:
                    p[k] = cal["p_minus"][k]
                cfg.append(p)
    Dm = die.measure(np.repeat(np.array(dac(cfg)), repeats, axis=0)).reshape(len(pairs), 2, 2, repeats).mean(-1)
    M = np.zeros((N, N))
    for i, (j, k) in enumerate(pairs):
        eps = -((Dm[i, 0, 1] - Dm[i, 0, 0]) - (Dm[i, 1, 1] - Dm[i, 1, 0])) / (2 * B[j])
        if abs(j - k) > reach:
            for jj in range(N):
                if abs(jj - k) > reach:
                    M[jj, k] = eps / dP[k]
        else:
            M[j, k] = eps / dP[k]
    return M, B


def calibrate_corrected(die, passes=3):
    """calibrate(), then the crosstalk map, so powers_for() can pre-correct every data pattern."""
    cal = calibrate(die, passes)
    M, B = measure_crosstalk(die, cal)
    cal.update(M=M, B=B, K=np.linalg.inv(np.diag(cal["g"]) + M))
    cal["lam"] = balance_weight(die, cal)
    return cal


def balance_weight(die, cal, repeats=16):
    """If the output coupler is not exactly 50/50, P+ - P- keeps a small term in S^2 and the line
    bends. Weighting P- by lam cancels it: lam is set so the readout of S = +16 and S = -16 averages
    to the readout of S = 0 (sixteen balanced patterns), all from the two output readings."""
    r = np.random.default_rng(1)
    zero = np.ones((16, N), int)
    for row in zero:
        row[r.choice(N, N // 2, replace=False)] = -1
    pats = np.vstack([np.ones((1, N), int), -np.ones((1, N), int), zero])
    pp, pm = die.measure_pm(np.repeat(powers_for(cal, pats), repeats, axis=0))
    pp, pm = pp.reshape(len(pats), repeats).mean(1), pm.reshape(len(pats), repeats).mean(1)
    curv = lambda P: (P[0] + P[1]) / 2 - P[2:].mean()  # noqa: E731
    return float(curv(pp) / curv(pm))


def uncorrected(cal):
    """The first calibration alone: no crosstalk map, plain difference of the two photocurrents."""
    return {k: v for k, v in cal.items() if k not in ("M", "K", "lam")}


def powers_for(cal, states):
    """states: (..., 16) in {+1, -1} (arm aligned with the reference or opposite) -> heater powers.
    With a crosstalk map in cal, each arm's heater is shifted so the heat its neighbours send it (and
    the reference) for this pattern is cancelled: (diag(g) + M) delta = -M dp."""
    s = np.asarray(states)
    pw = np.where(s > 0, cal["p_plus"][:N], cal["p_minus"][:N])
    if "K" in cal:
        e = (pw - cal["p_plus"][:N]) @ cal["M"].T
        pw = np.clip(pw - e @ cal["K"].T, 0, DAC_MAX)
    ref = np.broadcast_to(cal["p_plus"][N:], s.shape[:-1] + (1,))
    return np.concatenate([pw, ref], -1)


def all_states():
    k = np.arange(2 ** N)
    return 1 - 2 * ((k[:, None] >> np.arange(N)) & 1)


def sample_states(rng, per_level=470):
    """per_level random arm patterns for each of the 17 values of S (all of them at S = +-16)."""
    out = []
    for m in range(N + 1):                       # m arms at -1
        if m in (0, N):
            out.append(np.full((1, N), 1 - 2 * (m == N)))
            continue
        a = np.ones((per_level, N), int)
        for r in range(per_level):
            a[r, rng.choice(N, m, replace=False)] = -1
        out.append(a)
    return np.vstack(out)


def linearity(die, cal, states, full=True):
    D = readout(die.fields(powers_for(cal, states), full), cal)
    S = states.sum(1)
    A = np.vstack([S, np.ones_like(S)]).T
    (gain, off), *_ = np.linalg.lstsq(A, D, rcond=None)
    res = (D - (gain * S + off)) / (2 * gain)          # in levels
    return dict(gain_per_level=float(2 * gain), offset_levels=float(off / (2 * gain)), max_error_levels=float(np.abs(res).max()),
                rms_error_levels=float(res.std()), D=D, S=S, gain=gain, off=off)


def digit_inputs():
    """Every digit with 0, 1 and 2 pixels flipped: (images (n, 15) in {-1,+1}, true digit, flips)."""
    imgs, lab, fl = [], [], []
    for d, s in FONT.items():
        base = np.array([1 if ch == "1" else -1 for ch in s])
        imgs.append(base); lab.append(d); fl.append(0)
        for i in range(15):
            v = base.copy(); v[i] *= -1
            imgs.append(v); lab.append(d); fl.append(1)
        for i in range(15):
            for j in range(i + 1, 15):
                v = base.copy(); v[i] *= -1; v[j] *= -1
                imgs.append(v); lab.append(d); fl.append(2)
    return np.array(imgs), np.array(lab), np.array(fl)


def weights():
    W = np.zeros((10, N), int)
    for d, s in FONT.items():
        for i, ch in enumerate(s):
            W[d, ARM_OF_PIXEL[i]] = 1 if ch == "1" else -1
        W[d, BIAS_ARM] = 1
    return W


def arm_inputs(imgs):
    x = np.ones((imgs.shape[0], N), int)
    for i in range(15):
        x[:, ARM_OF_PIXEL[i]] = imgs[:, i]
    return x


def digits(die, cal, laser_W=1e-3, bw=1e6):
    """Each image is run against each stored digit (10 passes); the largest D wins."""
    imgs, lab, fl = digit_inputs()
    x = arm_inputs(imgs)
    W = weights()
    ideal = x @ W.T                                                     # (n, 10) scores
    states = x[:, None, :] * W[None, :, :]                              # (n, 10, 16)
    f = die.fields(powers_for(cal, states))
    Pp, Pm = laser_W * np.abs(f["plus"]) ** 2, laser_W * np.abs(f["minus"]) ** 2
    lam = cal.get("lam", 1.0)
    I = F.DET["resp"] * (Pp - lam * Pm)
    sig = np.sqrt((2 * F.P.QE * F.DET["resp"] * (Pp + lam ** 2 * Pm) + (1 + lam ** 2) * (F.DET["tia_pA"] * 1e-12) ** 2) * bw)
    I = I + sig * die.rng.standard_normal(I.shape)
    best = ideal.max(1, keepdims=True)
    unique = (ideal == best).sum(1) == 1                                # images with one clear nearest digit
    win_ideal = ideal.argmax(1)
    win_chip = I.argmax(1)
    agree = (win_chip == win_ideal)[unique]
    correct = (win_chip == lab)
    return dict(agree_with_ideal_pct=float(100 * agree.mean()), unique_images=int(unique.sum()), images=int(len(lab)),
                accuracy_pct={str(k): float(100 * correct[fl == k].mean()) for k in (0, 1, 2)},
                ideal_accuracy_pct={str(k): float(100 * (win_ideal == lab)[(fl == k)].mean()) for k in (0, 1, 2)},
                ties_ideal=int((~unique).sum()))


def linear_check(c, rng, n=400):
    phi = rng.uniform(0, 2 * np.pi, (n, len(ROLES)))
    full = c.fields({r: phi[:, j] for j, r in enumerate(ROLES)})
    lin = c.lin_fields(phi)
    scale = max(np.abs(full["plus"]).max(), np.abs(full["minus"]).max())
    return float(max(np.abs(full[o] - lin[o]).max() for o in full) / scale)


def path_delay(c, span=0.005):
    """Group delay of the signal path (all arms) minus the reference path, in ps, from the phase of
    signal / reference at the PLUS output across +-span um of wavelength."""
    wl = np.linspace(1.55 - span, 1.55 + span, 41)
    z = {r: np.zeros(wl.size) for r in ROLES}
    e0 = c.fields(z, wl=wl)["plus"]
    zr = dict(z)
    zr["REF"] = np.full(wl.size, np.pi)
    t_ref = (e0 - c.fields(zr, wl=wl)["plus"]) / 2
    w = 2 * np.pi * 299792458.0 / (wl * 1e-6)
    return float(np.polyfit(w, np.unwrap(np.angle((e0 - t_ref) / t_ref)), 1)[0] * 1e12)   # sax: fields go as exp(+i w t)


def spectral(c):
    """Phases set at 1550 nm (all arms in phase with the reference), then the wavelength swept."""
    wl = np.linspace(1.50, 1.60, 401)
    out = {}
    base = {r: 0.0 for r in ROLES}
    f0 = c.fields(base)
    # align every arm with the reference at 1550 nm using the linear model
    ref_ph = np.angle(c.t["plus"][-1] * np.conj(c.t["minus"][-1]))  # noqa: F841
    ph = np.zeros(len(ROLES))
    for j in range(N):
        th = np.linspace(0, 2 * np.pi, 721)
        P = np.zeros((th.size, len(ROLES))); P[:] = ph; P[:, j] = th
        ph[j] = th[np.argmax(diff_of(c.lin_fields(P)))]
    for name, flip in (("plus16", np.zeros(N)), ("minus16", np.full(N, np.pi))):
        p = ph.copy(); p[:N] += flip
        f = c.fields({r: np.full(wl.size, p[j]) for j, r in enumerate(ROLES)}, wl=wl)
        out[name] = diff_of(f)
    r = out["plus16"] / out["plus16"][np.argmin(np.abs(wl - 1.55))]
    del f0
    return dict({f"+-{d}nm_min_relative_swing": round(float(((out['plus16'] - out['minus16']) / (out['plus16'] - out['minus16'])[200])[np.abs(wl - 1.55) <= d * 1e-3 + 1e-9].min()), 3) for d in (1, 5, 10, 20, 50)},
                wavelength_nm=(wl * 1e3)[::4].round(1).tolist(), D_plus16=out["plus16"][::4].round(5).tolist(),
                D_minus16=out["minus16"][::4].round(5).tolist(), relative=r[::4].round(4).tolist())


def corner_study(args):
    """One process corner: n_dies dies, each calibrated in three steps, then measured."""
    corner, n_dies, seed = args
    X = crosstalk_matrix(geometry())
    c = Circuit(corner)
    lc = linear_check(c, np.random.default_rng(seed))
    S_all = all_states()
    rows, example = [], None
    for n in range(n_dies):
        die = Die(c, np.random.default_rng(seed * 1000 + int(corner[0]) * 10 + int(corner[1]) + n), X)
        cal = calibrate_corrected(die)
        st = sample_states(np.random.default_rng(n))
        lin = linearity(die, cal, st)
        lin_u = linearity(die, uncorrected(cal), st)
        ex = linearity(die, cal, S_all, full=False)
        dl = readout(die.fields(powers_for(cal, st), full=False), cal)
        dg = digits(die, cal)
        rows.append(dict(max_error_levels=lin["max_error_levels"], rms_error_levels=lin["rms_error_levels"],
                         uncorrected_max_error_levels=lin_u["max_error_levels"], balance_weight=cal["lam"],
                         exhaustive_linear_model_max_error_levels=ex["max_error_levels"],
                         linear_model_error_levels=float(np.abs(dl - lin["D"]).max() / lin["gain_per_level"]),
                         gain_uW_per_level_at_1mW=lin["gain_per_level"] * 1e3, digits=dg))
        if n == 0:
            example = dict(points_uW_per_mW={str(S): (lin["D"][lin["S"] == S][:60] * 1e3).round(4).tolist() for S in range(-N, N + 1, 2)},
                           gain_uW_per_level_at_1mW=lin["gain_per_level"] * 1e3, max_error_levels=lin["max_error_levels"],
                           uncorrected_max_error_levels=lin_u["max_error_levels"], balance_weight=cal["lam"],
                           calibration_mW=dict(plus=(cal["p_plus"] * 1e3).round(3).tolist(), minus=(cal["p_minus"] * 1e3).round(3).tolist()))
        print(corner, n, round(lin_u["max_error_levels"], 3), "->", round(lin["max_error_levels"], 3), flush=True)
    key = f"{int(corner[0])}nm_x_{int(corner[1])}nm"
    r = lambda k: [row[k] for row in rows]  # noqa: E731
    return key, dict(
        dies=n_dies, linear_model_error=lc,
        worst_max_error_levels=round(max(r("max_error_levels")), 3),
        median_max_error_levels=round(float(np.median(r("max_error_levels"))), 3),
        worst_uncorrected_max_error_levels=round(max(r("uncorrected_max_error_levels")), 3),
        median_uncorrected_max_error_levels=round(float(np.median(r("uncorrected_max_error_levels"))), 3),
        worst_rms_error_levels=round(max(r("rms_error_levels")), 4),
        balance_weight_range=[round(min(r("balance_weight")), 3), round(max(r("balance_weight")), 3)],
        worst_exhaustive_linear_model_max_error_levels=round(max(r("exhaustive_linear_model_max_error_levels")), 3),
        worst_linear_model_error_levels=round(max(r("linear_model_error_levels")), 3),
        level_uW_at_1mW=round(float(np.median(r("gain_uW_per_level_at_1mW"))), 2),
        worst_agree_with_ideal_pct=round(min(row["digits"]["agree_with_ideal_pct"] for row in rows), 2),
        digits_example=rows[0]["digits"]), example


def main(n_dies=20, seed=11, procs=2):
    import multiprocessing as mp
    X = crosstalk_matrix(geometry())
    res = {"method": __doc__.strip().splitlines()[0], "netlist": os.path.relpath(EXTRACT, HERE),
           "die_model": dict(P_pi_mW=P_PI * 1e3, efficiency_spread=ETA_SPREAD, crosstalk=f"{XT0:.0%} at {XT_D0:.0f} um, decay length {XT_L:.0f} um",
                             nearest_neighbour_crosstalk=round(float(X[1, 0]), 4), measurement_noise=MEAS_NOISE,
                             dac="16-bit, 0-80 mW"),
           "font": FONT, "arm_of_pixel": ARM_OF_PIXEL, "bias_arm": BIAS_ARM}
    with mp.get_context("spawn").Pool(procs) as pool:
        out = pool.map(corner_study, [(c, n_dies, seed) for c in F.CORNERS], chunksize=1)
    res["corners"] = {k: v for k, v, _ in out}
    res["example_nominal"] = next(e for k, _, e in out if k == "220nm_x_500nm")
    for k, v in res["corners"].items():
        print(k, "uncorrected", v["worst_uncorrected_max_error_levels"], "corrected", v["worst_max_error_levels"], "digits", v["worst_agree_with_ideal_pct"])
    c = Circuit()
    die0 = Die(c, np.random.default_rng(seed), np.eye(len(ROLES)))
    cal0 = calibrate_corrected(die0)
    res["example_nominal"]["max_error_levels_without_crosstalk"] = linearity(die0, cal0, sample_states(np.random.default_rng(0)))["max_error_levels"]
    # noise: the laser power at which one level is 7 sigma at 1 MHz (heater chip) and at 10 and 50 GHz detection
    g = res["example_nominal"]["gain_uW_per_level_at_1mW"] * 1e-3      # per W of laser
    lp = {}
    for bw in (1e6, 1e10, 5e10):
        lo, hi = 1e-9, 1.0
        for _ in range(80):
            pl = math.sqrt(lo * hi)
            I1 = F.DET["resp"] * pl * g
            sig = math.sqrt((2 * F.P.QE * F.DET["resp"] * pl * 0.5 + 2 * (F.DET["tia_pA"] * 1e-12 if bw < 1e9 else 20e-12) ** 2) * bw)
            lo, hi = (pl, hi) if I1 / (2 * sig) < 7 else (lo, pl)
        lp[f"{bw:.0e}Hz"] = round(hi * 1e6, 2)
    res["laser_uW_for_one_level_at_Q7"] = lp
    res["path_delay_signal_minus_reference_ps"] = path_delay(c)
    res["spectral"] = spectral(c)
    res["digit_distances"] = {f"{a}-{b}": int(sum(x != y for x, y in zip(FONT[a], FONT[b]))) for a in FONT for b in FONT if a < b}
    with open(os.path.join(HERE, "perceptron_results.json"), "w") as f:
        json.dump(res, f, indent=1, default=float)
    print("no crosstalk:", res["example_nominal"]["max_error_levels_without_crosstalk"], "laser:", lp, "delay ps:", res["path_delay_signal_minus_reference_ps"])
    print("spectral", {k: v for k, v in res["spectral"].items() if k.startswith("+-")})


if __name__ == "__main__":
    main()
