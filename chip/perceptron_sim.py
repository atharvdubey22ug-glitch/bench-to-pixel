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
drivers): each arm's heater is swept in turn and set where the difference signal is largest (the
arm in phase with the reference = +1); -1 is half a period away. Three passes, because each heater
also warms its neighbours. In use, the arms switch between the two settings with the data, and the
crosstalk that changes with the data is left uncorrected.

Results (perceptron_results.json):
  linearity    D against S on 20 dies per corner: 7,522 arm patterns (470 for each value of S) through the
               full model, and all 65,536 through the linear model
  digits       a 3x5-pixel digit font as stored weight patterns (one pass of light per digit); every
               digit with 0, 1 and 2 pixels flipped, the winner read from the chip's 10 outputs
  spectral     D for S = +16 and -16 from 1500 to 1600 nm
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


def diff_of(f):
    return np.abs(f["plus"]) ** 2 - np.abs(f["minus"]) ** 2


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


def powers_for(cal, states):
    """states: (..., 16) in {+1, -1} (arm aligned with the reference or opposite) -> heater powers."""
    s = np.asarray(states)
    pw = np.where(s > 0, cal["p_plus"][:N], cal["p_minus"][:N])
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
    D = diff_of(die.fields(powers_for(cal, states), full))
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
    I = F.DET["resp"] * (Pp - Pm)
    sig = np.sqrt((2 * F.P.QE * F.DET["resp"] * (Pp + Pm) + 2 * (F.DET["tia_pA"] * 1e-12) ** 2) * bw)
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


def main(n_dies=20, seed=11):
    pos = geometry()
    X = crosstalk_matrix(pos)
    res = {"method": __doc__.strip().splitlines()[0], "netlist": os.path.relpath(EXTRACT, HERE),
           "die_model": dict(P_pi_mW=P_PI * 1e3, efficiency_spread=ETA_SPREAD, crosstalk=f"{XT0:.0%} at {XT_D0:.0f} um, decay length {XT_L:.0f} um",
                             nearest_neighbour_crosstalk=round(float(X[1, 0]), 4), measurement_noise=MEAS_NOISE,
                             dac="16-bit, 0-80 mW"),
           "font": FONT, "arm_of_pixel": ARM_OF_PIXEL, "bias_arm": BIAS_ARM}
    rng = np.random.default_rng(seed)
    S_all = all_states()
    res["corners"], examples = {}, {}
    for corner in F.CORNERS:
        c = Circuit(corner)
        lc = linear_check(c, rng)
        rows = []
        for n in range(n_dies):
            die = Die(c, np.random.default_rng(seed * 1000 + int(corner[0]) * 10 + int(corner[1]) + n), X)
            cal = calibrate(die)
            st = sample_states(np.random.default_rng(n))
            lin = linearity(die, cal, st)
            ex = linearity(die, cal, S_all, full=False)
            dl = diff_of(die.fields(powers_for(cal, st), full=False))
            dg = digits(die, cal)
            rows.append(dict(max_error_levels=lin["max_error_levels"], rms_error_levels=lin["rms_error_levels"],
                             exhaustive_linear_model_max_error_levels=ex["max_error_levels"],
                             linear_model_error_levels=float(np.abs(dl - lin["D"]).max() / lin["gain_per_level"]),
                             gain_uW_per_level_at_1mW=lin["gain_per_level"] * 1e3, digits=dg))
            if n == 0:
                examples[corner] = (die, cal, lin)
        key = f"{int(corner[0])}nm_x_{int(corner[1])}nm"
        res["corners"][key] = dict(
            dies=n_dies, linear_model_error=lc,
            worst_max_error_levels=round(max(r["max_error_levels"] for r in rows), 3),
            median_max_error_levels=round(float(np.median([r["max_error_levels"] for r in rows])), 3),
            worst_rms_error_levels=round(max(r["rms_error_levels"] for r in rows), 4),
            worst_exhaustive_linear_model_max_error_levels=round(max(r["exhaustive_linear_model_max_error_levels"] for r in rows), 3),
            worst_linear_model_error_levels=round(max(r["linear_model_error_levels"] for r in rows), 3),
            level_uW_at_1mW=round(float(np.median([r["gain_uW_per_level_at_1mW"] for r in rows])), 2),
            worst_agree_with_ideal_pct=round(min(r["digits"]["agree_with_ideal_pct"] for r in rows), 2),
            digits_example=rows[0]["digits"])
        print(key, res["corners"][key]["worst_max_error_levels"], res["corners"][key]["worst_agree_with_ideal_pct"], f"lin {lc:.1e}", flush=True)
    # example die at the nominal corner: the D vs S cloud, its calibration, and a no-crosstalk comparison
    die, cal, lin = examples[F.NOMINAL]
    cl = {}
    for S in range(-N, N + 1, 2):
        m = lin["S"] == S
        cl[str(S)] = [float(lin["D"][m].min() * 1e6), float(lin["D"][m].max() * 1e6)]   # uW per W of laser = 1e6 * D ... per W
    res["example_nominal"] = dict(D_range_by_S_uW_per_mW={k: [v[0] / 1e3, v[1] / 1e3] for k, v in cl.items()},
                                  gain_uW_per_level_at_1mW=lin["gain_per_level"] * 1e6 / 1e3, max_error_levels=lin["max_error_levels"],
                                  points_uW_per_mW={str(S): (lin["D"][lin["S"] == S][:60] * 1e3).round(4).tolist() for S in range(-N, N + 1, 2)},
                                  calibration_mW=dict(plus=(cal["p_plus"] * 1e3).round(3).tolist(), minus=(cal["p_minus"] * 1e3).round(3).tolist()))
    die0 = Die(die.c, np.random.default_rng(seed), np.eye(len(ROLES)))
    cal0 = calibrate(die0)
    res["example_nominal"]["max_error_levels_without_crosstalk"] = linearity(die0, cal0, sample_states(np.random.default_rng(0)))["max_error_levels"]
    # noise: the laser power at which one level is 7 sigma at 1 MHz (heater chip) and at 10 GHz detection
    g = lin["gain_per_level"]
    out = {}
    for bw in (1e6, 1e10, 5e10):
        lo, hi = 1e-9, 1.0
        for _ in range(80):
            pl = math.sqrt(lo * hi)
            I1 = F.DET["resp"] * pl * g
            sig = math.sqrt((2 * F.P.QE * F.DET["resp"] * pl * 0.5 + 2 * (F.DET["tia_pA"] * 1e-12 if bw < 1e9 else 20e-12) ** 2) * bw)
            lo, hi = (pl, hi) if I1 / (2 * sig) < 7 else (lo, pl)
        out[f"{bw:.0e}Hz"] = round(hi * 1e6, 2)
    res["laser_uW_for_one_level_at_Q7"] = out
    res["spectral"] = spectral(Circuit())
    res["digit_distances"] = {f"{a}-{b}": int(sum(x != y for x, y in zip(FONT[a], FONT[b]))) for a in FONT for b in FONT if a < b}
    with open(os.path.join(HERE, "perceptron_results.json"), "w") as f:
        json.dump(res, f, indent=1, default=float)
    print("no crosstalk:", res["example_nominal"]["max_error_levels_without_crosstalk"], "laser:", out)
    print("spectral", {k: v for k, v in res["spectral"].items() if k.startswith("+-")})


if __name__ == "__main__":
    main()
