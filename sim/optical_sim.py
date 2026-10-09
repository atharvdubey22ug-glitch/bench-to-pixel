#!/usr/bin/env python3
"""
Circuit-level simulation for Bench to Pixel.

Three experiments:
  1. The half-adder cell: the paper's bench layout (Fig. 14) as a silicon
     photonic circuit. Truth-table photocurrents, decision margins, bit error
     rate, the XOR bias-phase tolerance and the minimum laser power.
  2. A 4x4 optical dot-product tile: one WDM channel per input row, ring
     modulators as inputs, tunable couplers as weights, photodiodes summing
     each column. Effective bits versus laser power and symbol rate.
  3. The lighting dot product N.L on that tile (offset-encoded inputs,
     balanced columns for signed weights). Produces the parameters the
     browser model on the page uses, plus a check that the browser model
     reproduces this simulation.

Method: analytical compact models (component transfer functions) with shot,
transimpedance-amplifier and laser intensity noise. There is no full-wave
electromagnetic simulation. Every number in PARAMS is an assumption chosen as
a typical value for a silicon photonics foundry process at 1550 nm.

Run:  python3 optical_sim.py      (writes results.json and figures/*.png)
"""
import json
import math
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
q = 1.602176634e-19          # C
c = 299_792_458.0            # m/s
LAM = 1550e-9                # m

PARAMS = {
    # passive components
    "mmi_excess_dB": 0.2,         # per pass through an MMI
    "mmi_imbalance_dB": 0.1,      # power ratio between the two MMI outputs
    "bend_dB": 0.02,              # per 90-degree bend (the bench's mirrors 1, 2)
    "loop_mirror_dB": 0.3,        # Sagnac loop mirror excess loss (mirrors 9, 10)
    "arm_mismatch_dB": 0.1,       # loss difference between the two Michelson arms
    "coupler_through_dB": 0.1,    # tunable-coupler through-port loss in the tile
    "crossing_dB": 0.15,          # waveguide crossing loss in the tile
    # ring modulator (carrier depletion)
    "ring_radius_um": 5.0,
    "group_index": 4.2,
    "ring_loss_dB_per_cm": 40.0,  # doped ring, round-trip loss
    "ring_self_coupling_r": 0.983,
    "mod_efficiency_pm_per_V": 40.0,
    "mod_swing_V": 2.0,
    "ring_thermal_pm_per_K": 70.0,
    # detection
    "pd_responsivity_A_per_W": 0.9,
    "pd_dark_A": 20e-9,
    "tia_noise_A_per_rtHz": 20e-12,
    "laser_RIN_dB_per_Hz": -140.0,
    # converters and calibration (tile)
    "input_dac_bits": 6,
    "heater_dac_bits": 8,
    "weight_calibration_sigma": 0.003,   # residual weight error after calibration
}
PR = PARAMS
R_PD = PR["pd_responsivity_A_per_W"]


def dB(x):
    """dB loss -> power transmission."""
    return 10 ** (-x / 10)


# ---------------------------------------------------------------- ring modulator
L_RING = 2 * math.pi * PR["ring_radius_um"] * 1e-6
A_RING = 10 ** (-(PR["ring_loss_dB_per_cm"] * 100) * L_RING / 20)   # round-trip field amplitude
R_RING = PR["ring_self_coupling_r"]


def ring_t(dlam):
    """All-pass ring field transmission at detuning dlam = lambda - lambda_resonance (m)."""
    phi = 2 * np.pi * PR["group_index"] * L_RING * np.asarray(dlam, dtype=float) / LAM ** 2
    e = A_RING * np.exp(-1j * phi)
    return (R_RING - e) / (1 - R_RING * e)


def ring_summary():
    ra = R_RING * A_RING
    fwhm = (1 - ra) * LAM ** 2 / (math.pi * PR["group_index"] * L_RING * math.sqrt(ra))
    fsr = LAM ** 2 / (PR["group_index"] * L_RING)
    swing = PR["mod_efficiency_pm_per_V"] * PR["mod_swing_V"] * 1e-12
    t0, t1 = abs(ring_t(0.0)) ** 2, abs(ring_t(swing)) ** 2
    return {
        "round_trip_amplitude_a": A_RING,
        "fwhm_pm": fwhm * 1e12,
        "loaded_Q": LAM / fwhm,
        "photon_lifetime_bandwidth_GHz": c * fwhm / LAM ** 2 / 1e9,
        "fsr_nm": fsr * 1e9,
        "T_on_resonance": t0,
        "T_detuned_by_swing": t1,
        "binary_extinction_dB": 10 * math.log10(t1 / t0),
        "binary_insertion_loss_dB": -10 * math.log10(t1),
    }


def noise_sigma(i_mean, bandwidth, optical_powers=()):
    """Detector current noise: shot + TIA + laser RIN (one independent RIN term per beam)."""
    shot = 2 * q * (np.asarray(i_mean) + PR["pd_dark_A"]) * bandwidth
    tia = PR["tia_noise_A_per_rtHz"] ** 2 * bandwidth
    rin = 10 ** (PR["laser_RIN_dB_per_Hz"] / 10) * bandwidth * sum((R_PD * np.asarray(p)) ** 2 for p in optical_powers)
    return np.sqrt(shot + tia + rin)


def q_to_ber(qf):
    return 0.5 * math.erfc(qf / math.sqrt(2))


# ======================================================= 1. half-adder cell
SWING = PR["mod_efficiency_pm_per_V"] * PR["mod_swing_V"] * 1e-12


def half_adder(p0=1e-3, rate=1e9, phase_err_deg=0.0):
    """Photocurrents for the four input pairs. Bit 0 = ring on resonance (beam blocked),
    bit 1 = ring detuned by the full drive swing (beam passes)."""
    bw = rate / 2
    eta = dB(PR["mmi_excess_dB"])
    imb = 10 ** (PR["mmi_imbalance_dB"] / 20)
    tap, thru = eta * 0.5 * imb, eta * 0.5 / imb              # 1x2 MMI outputs
    pa_pre = p0 * tap * dB(2 * PR["bend_dB"])                  # S1 tap, two bends (S1 junction, mirror 1)
    pb_pre = p0 * thru * tap * dB(2 * PR["bend_dB"])           # S2 tap
    p3 = p0 * thru * thru                                     # into S3 (Michelson)
    # intensity marking: carry levels A alone : B alone = 2 : 3 (the paper's distinct intensities)
    w_b = 1.0
    w_a = (2 / 3) * pb_pre / pa_pre
    t = {0: ring_t(0.0), 1: ring_t(SWING)}
    r_loop = math.sqrt(dB(PR["loop_mirror_dB"]))
    mismatch = math.sqrt(dB(PR["arm_mismatch_dB"]))
    bias = np.pi + np.deg2rad(phase_err_deg)                   # heater sets the dark output
    rows = []
    for a in (0, 1):
        for b in (0, 1):
            pc_a = pa_pre * w_a * abs(t[a]) ** 2
            pc_b = pb_pre * w_b * abs(t[b]) ** 2
            i_c = R_PD * (pc_a + pc_b) + PR["pd_dark_A"]
            rho_a = t[a] ** 2 * r_loop                         # double pass through the arm's ring
            rho_b = t[b] ** 2 * r_loop * mismatch * np.exp(1j * bias)
            p_s = eta ** 2 * 0.25 * abs(rho_a + rho_b) ** 2 * p3   # 2x2 MMI: tau*kappa = 1/4
            i_s = R_PD * p_s + PR["pd_dark_A"]
            rows.append({
                "a": a, "b": b,
                "carry_uA": i_c * 1e6, "sum_uA": i_s * 1e6,
                "carry_sigma_uA": float(noise_sigma(i_c, bw, (pc_a, pc_b))) * 1e6,
                "sum_sigma_uA": float(noise_sigma(i_s, bw, (p_s,))) * 1e6,
            })
    return rows, {"w_a": w_a, "w_b": w_b, "p_into_michelson_mW": p3 * 1e3}


def decide(rows):
    """Thresholds midway between the closest 0 and 1 levels, and the Q factor of each decision."""
    def level(key, logic):
        return [(r[key + "_uA"], r[key + "_sigma_uA"]) for r in rows if logic(r["a"], r["b"])]
    out = {}
    for key, fn in (("carry", lambda a, b: a & b), ("sum", lambda a, b: a ^ b)):
        ones = level(key, lambda a, b: fn(a, b) == 1)
        zeros = level(key, lambda a, b: fn(a, b) == 0)
        hi0 = max(zeros, key=lambda z: z[0])
        lo1 = min(ones, key=lambda o: o[0])
        thr = (hi0[0] + lo1[0]) / 2
        qf = (lo1[0] - hi0[0]) / (lo1[1] + hi0[1])
        out[key] = {"threshold_uA": thr, "Q": qf, "BER": q_to_ber(qf),
                    "worst_zero_uA": hi0[0], "weakest_one_uA": lo1[0]}
    out["min_Q"] = min(out["carry"]["Q"], out["sum"]["Q"])
    return out


def half_adder_report():
    rows, design = half_adder()
    dec = decide(rows)
    # XOR bias-phase sweep: both arms open (should be 0) and one arm open (1)
    phases = np.linspace(-90, 90, 181)
    both, one = [], []
    for ph in phases:
        r, _ = half_adder(phase_err_deg=ph)
        both.append(r[3]["sum_uA"])
        one.append(r[2]["sum_uA"])
    thr = dec["sum"]["threshold_uA"]
    sig = rows[3]["sum_sigma_uA"]
    ok = [abs(ph) for ph, i0 in zip(phases, both) if (thr - i0) / sig >= 7.0]
    tol = 0.0
    for ph, i0 in zip(phases, both):          # contiguous band around 0 with Q >= 7
        if ph >= 0 and (thr - i0) / sig >= 7.0:
            tol = ph
        elif ph >= 0:
            break
    # minimum laser power for Q >= 7 on every decision
    lo, hi = 1e-7, 1e-3
    for _ in range(60):
        mid = math.sqrt(lo * hi)
        if decide(half_adder(p0=mid)[0])["min_Q"] >= 7:
            hi = mid
        else:
            lo = mid
    # thermal: phase drift per kelvin of temperature difference between 100-um arms
    dndT = 1.86e-4
    deg_per_K = math.degrees(2 * math.pi * dndT * 2 * 100e-6 / LAM)
    return {
        "laser_mW": 1.0, "rate_Gbps": 1.0,
        "design": design, "levels": rows, "decisions": dec,
        "xor_phase_sweep": {"phase_deg": phases.tolist(), "both_open_uA": both, "one_open_uA": one,
                            "threshold_uA": thr},
        "xor_phase_tolerance_deg": tol,
        "min_laser_power_uW_for_Q7": hi * 1e6,
        "arm_phase_drift_deg_per_K_for_100um_arms": deg_per_K,
        "xor_extinction_dB": 10 * math.log10(rows[2]["sum_uA"] / max(rows[3]["sum_uA"] - PR["pd_dark_A"] * 1e6, 1e-9)),
    }


# ======================================================= 2. 4x4 tile
def _mod_window():
    """Place the drive window on the ring's slope where it gives the widest transmission range."""
    best = (-1, 0.0)
    for d0 in np.linspace(0, 400e-12, 801):
        span = abs(ring_t(d0 + SWING)) ** 2 - abs(ring_t(d0)) ** 2
        if span > best[0]:
            best = (span, d0)
    return best[1]


D0 = _mod_window()
DGRID = np.linspace(D0, D0 + SWING, 4001)
TGRID = np.abs(ring_t(DGRID)) ** 2
T_LO, T_HI = float(TGRID[0]), float(TGRID[-1])
SHIFT_PER_K = PR["ring_thermal_pm_per_K"] * 1e-12


def encode(x, d_t=0.0, dac=True):
    """Input value x in [0,1] -> ring power transmission, with DAC-quantised predistortion and drift."""
    target = T_LO + np.clip(x, 0, 1) * (T_HI - T_LO)
    d = np.interp(target, TGRID, DGRID)
    if dac:
        levels = 2 ** PR["input_dac_bits"] - 1
        d = D0 + np.round((d - D0) / SWING * levels) / levels * SWING
    return np.abs(ring_t(d - SHIFT_PER_K * d_t)) ** 2


T_THRU, T_CROSS = dB(PR["coupler_through_dB"]), dB(PR["crossing_dB"])


def _gain():
    """Largest common weight scale g the crossbar can realise for an all-ones weight matrix."""
    def feasible(g):
        for j in range(4):
            rem = 1.0
            for i in range(4):
                d = g / (rem * T_THRU ** i * T_CROSS ** (3 - j))
                if d > 1:
                    return False
                rem *= 1 - d
        return True
    lo, hi = 0.0, 1.0
    for _ in range(60):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if feasible(mid) else (lo, mid)
    return lo


G = _gain()


def realise(w, rng, quant=True, cal=True):
    """Target weights w[..., i(out), j(in)] in [0,1] -> realised weights after heater DAC and calibration."""
    w = np.asarray(w, dtype=float)
    out = np.empty_like(w)
    levels = 2 ** PR["heater_dac_bits"] - 1
    for j in range(w.shape[-1]):
        rem = np.ones(w.shape[:-2])
        for i in range(w.shape[-2]):
            loss = T_THRU ** i * T_CROSS ** (3 - j)
            d = np.clip(G * w[..., i, j] / (rem * loss), 0, 1)
            if quant:
                theta = 2 * np.arcsin(np.sqrt(d))
                d = np.sin(np.round(theta / np.pi * levels) / levels * np.pi / 2) ** 2
            out[..., i, j] = d * rem * loss / G
            rem = rem * (1 - d)
    if cal:
        out = out + rng.normal(0, PR["weight_calibration_sigma"], out.shape)
    return out


def tile_mc(p_ch, rate, n=12000, adc_bits=10, d_t=0.0, rng=None, impair=("dac", "weights", "noise", "adc")):
    rng = rng or np.random.default_rng(1)
    bw = rate / 2
    x = rng.random((n, 4))
    w = rng.random((n, 4, 4))                        # [sample, out i, in j]
    y_true = np.einsum("nij,nj->ni", w, x) / 4
    t = encode(x, d_t, dac="dac" in impair)
    w_r = realise(w, rng, quant="weights" in impair, cal="weights" in impair)
    p_ij = p_ch * t[:, None, :] * G * w_r             # optical power reaching each column PD per row
    i_mean = R_PD * p_ij.sum(-1) + PR["pd_dark_A"]
    if "noise" in impair:
        rin = 10 ** (PR["laser_RIN_dB_per_Hz"] / 10) * bw * ((R_PD * p_ij) ** 2).sum(-1)
        sig = np.sqrt(noise_sigma(i_mean, bw) ** 2 + rin)
        i_meas = i_mean + sig * rng.standard_normal(i_mean.shape)
    else:
        i_meas = i_mean
    offset = R_PD * p_ch * T_LO * G * w.sum(-1) + PR["pd_dark_A"]
    y = (i_meas - offset) / (R_PD * p_ch * (T_HI - T_LO) * G * 4)
    if "adc" in impair:
        lv = 2 ** adc_bits - 1
        y = np.round(np.clip(y, 0, 1) * lv) / lv
    err = y - y_true
    s = float(np.std(err))
    return s, -math.log2(math.sqrt(12) * s)


def tile_report():
    rng = np.random.default_rng(3)
    powers = np.logspace(-2, math.log10(5), 22)       # mW per channel
    sweep = {}
    for rate in (1e9, 5e9, 10e9):
        sweep[f"{rate / 1e9:g}"] = [tile_mc(p * 1e-3, rate, rng=rng)[1] for p in powers]
    design = dict(p_mW=0.5, rate_GSps=5, adc_bits=8)
    full = tile_mc(0.5e-3, 5e9, adc_bits=8, rng=rng)
    budget = {}
    for name in ("noise", "dac", "weights", "adc"):
        budget[name] = tile_mc(0.5e-3, 5e9, adc_bits=8, rng=rng, impair=(name,))[1]
    return {
        "channels": 4, "channel_spacing_GHz": 200, "weight_scale_g": G,
        "input_T_range": [T_LO, T_HI],
        "power_mW": powers.tolist(), "enob_by_rate": sweep,
        "design_point": design, "design_enob": full[1], "design_rms_error": full[0],
        "enob_with_one_impairment": budget,
        "throughput_note": "16 multiply-accumulates per symbol per tile",
    }


# ======================================================= 3. lighting N.L on the tile
SQ3 = math.sqrt(3)


def unit_vectors(rng, n):
    v = rng.normal(size=(n, 3))
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def lighting_circuit(p_ch, rate, n=20000, adc_bits=8, d_t=0.0, rng=None, impair=("dac", "weights", "noise", "adc")):
    """Full circuit model: offset-encoded normals x=(N+1)/2, signed light vector on two balanced columns."""
    rng = rng or np.random.default_rng(5)
    bw = rate / 2
    nrm, lgt = unit_vectors(rng, n), unit_vectors(rng, n)
    x = (nrm + 1) / 2
    lp, lm = np.maximum(lgt, 0), np.maximum(-lgt, 0)
    t = encode(x, d_t, dac="dac" in impair)
    w = np.stack([lp, lm], axis=1)                                  # [n, 2 columns, 3 inputs]
    w4 = np.concatenate([w, np.zeros((n, 2, 1))], axis=2)
    w4 = np.concatenate([w4, np.zeros((n, 2, 4))], axis=1)          # pad to 4x4
    w_r = realise(w4, rng, quant="weights" in impair, cal="weights" in impair)[:, :2, :3]
    p_ij = p_ch * t[:, None, :] * G * w_r
    i_mean = R_PD * p_ij.sum(-1) + PR["pd_dark_A"]
    if "noise" in impair:
        rin = 10 ** (PR["laser_RIN_dB_per_Hz"] / 10) * bw * ((R_PD * p_ij) ** 2).sum(-1)
        sig = np.sqrt(noise_sigma(i_mean, bw) ** 2 + rin)
        i_meas = i_mean + sig * rng.standard_normal(i_mean.shape)
    else:
        i_meas = i_mean
    offset = R_PD * p_ch * T_LO * G * w.sum(-1) + PR["pd_dark_A"]
    scale = R_PD * p_ch * (T_HI - T_LO) * G
    y = ((i_meas[:, 0] - offset[:, 0]) - (i_meas[:, 1] - offset[:, 1])) / scale
    if "adc" in impair:
        lv = 2 ** adc_bits - 1
        y = np.round((np.clip(y, -SQ3, SQ3) + SQ3) / (2 * SQ3) * lv) / lv * 2 * SQ3 - SQ3
    nl = 2 * y - lgt.sum(1)
    err = nl - (nrm * lgt).sum(1)
    return float(np.std(err))


def drift_poly(d_t):
    """Encoded-input error after a ring temperature error d_t: x' - x fitted with a quadratic in x."""
    x = np.linspace(0, 1, 201)
    xp = (encode(x, d_t, dac=False) - T_LO) / (T_HI - T_LO)
    c2, c1, c0 = np.polyfit(x, xp - x, 2)
    return [float(c0), float(c1), float(c2)]


def browser_model(sig_y, coeffs, adc_bits, rng, n=20000, sigma_w=0.0, dac_bits=6):
    """The simplified model the page runs per pixel (mirrors the shader)."""
    nrm, lgt = unit_vectors(rng, n), unit_vectors(rng, n)
    x = (nrm + 1) / 2
    lv_in = 2 ** dac_bits - 1
    x = np.round(x * lv_in) / lv_in
    c0, c1, c2 = coeffs
    x = x + c0 + c1 * x + c2 * x * x
    lw = lgt + rng.normal(0, sigma_w, lgt.shape) * 1.0
    y = (lw * x).sum(1) + sig_y * rng.standard_normal(n)
    lv = 2 ** adc_bits - 1
    y = np.round((np.clip(y, -SQ3, SQ3) + SQ3) / (2 * SQ3) * lv) / lv * 2 * SQ3 - SQ3
    nl = 2 * y - lgt.sum(1)
    return float(np.std(nl - (nrm * lgt).sum(1)))


def lighting_report():
    rng = np.random.default_rng(11)
    rate = 5e9
    powers = np.logspace(-2, math.log10(5), 22)
    # noise-only sigma of y (balanced output, in weight*x units), per laser power
    sig_y = [lighting_circuit(p * 1e-3, rate, rng=rng, impair=("noise",)) / 2 for p in powers]
    temps = np.linspace(0, 0.5, 26)
    drift = [drift_poly(t) for t in temps]
    # realised-weight error (heater quantisation + calibration), in weight units
    w = rng.random((4000, 4, 4))
    sigma_w = float(np.std(realise(w, rng) - w))
    full_default = lighting_circuit(0.5e-3, rate, rng=rng)
    sy = float(np.interp(math.log10(0.5), np.log10(powers), sig_y))
    browser_default = browser_model(sy, drift_poly(0.0), 8, rng, sigma_w=sigma_w)
    full_warm = lighting_circuit(0.5e-3, rate, rng=rng, d_t=0.1)
    browser_warm = browser_model(sy, drift_poly(0.1), 8, rng, sigma_w=sigma_w)
    return {
        "rate_GSps": 5, "power_mW": powers.tolist(), "sigma_y": sig_y,
        "drift_dT_K": temps.tolist(), "drift_coeffs": drift, "sigma_w": sigma_w,
        "dac_bits": PR["input_dac_bits"], "default": {"p_mW": 0.5, "adc_bits": 8, "dT_K": 0.0},
        "validation": {
            "circuit_rms_NL_error_default": full_default, "browser_rms_NL_error_default": browser_default,
            "circuit_rms_NL_error_dT_0.1K": full_warm, "browser_rms_NL_error_dT_0.1K": browser_warm,
        },
    }


# ======================================================= figures
def figures(res):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    os.makedirs(os.path.join(HERE, "figures"), exist_ok=True)
    ink, fg, mut = "#0a1013", "#e3ebed", "#8b9ea5"
    plt.rcParams.update({"figure.facecolor": ink, "axes.facecolor": ink, "axes.edgecolor": "#33464f",
                         "axes.labelcolor": mut, "xtick.color": mut, "ytick.color": mut, "text.color": fg,
                         "font.size": 10, "axes.grid": True, "grid.color": "#1c2a31"})
    sw = res["half_adder"]["xor_phase_sweep"]
    fig, ax = plt.subplots(figsize=(6, 3.4))
    ax.plot(sw["phase_deg"], sw["one_open_uA"], color="#ff5a47", label="one arm open (sum = 1)")
    ax.plot(sw["phase_deg"], sw["both_open_uA"], color="#8fe3ff", label="both arms open (sum = 0)")
    ax.axhline(sw["threshold_uA"], color=mut, ls="--", lw=1, label="threshold")
    tol = res["half_adder"]["xor_phase_tolerance_deg"]
    ax.axvspan(-tol, tol, color="#eba340", alpha=.12, label=f"±{tol:.0f}° keeps Q ≥ 7")
    ax.set_xlabel("XOR bias phase error (degrees)"); ax.set_ylabel("sum photocurrent (µA)")
    ax.legend(frameon=False, fontsize=8); fig.tight_layout()
    fig.savefig(os.path.join(HERE, "figures", "xor_phase_tolerance.png"), dpi=160); plt.close(fig)
    t = res["tile"]
    fig, ax = plt.subplots(figsize=(6, 3.4))
    for k, col in zip(("1", "5", "10"), ("#eba340", "#ff5a47", "#8fe3ff")):
        ax.semilogx(t["power_mW"], t["enob_by_rate"][k], color=col, label=f"{k} GS/s")
    ax.set_xlabel("laser power per WDM channel (mW)"); ax.set_ylabel("effective bits")
    ax.legend(frameon=False, fontsize=8); fig.tight_layout()
    fig.savefig(os.path.join(HERE, "figures", "tile_effective_bits.png"), dpi=160); plt.close(fig)


def main():
    res = {
        "assumptions": PARAMS,
        "ring": ring_summary(),
        "half_adder": half_adder_report(),
        "tile": tile_report(),
        "lighting": lighting_report(),
    }
    with open(os.path.join(HERE, "results.json"), "w") as f:
        json.dump(res, f, indent=1)
    figures(res)
    return res


if __name__ == "__main__":
    r = main()
    ha, tl, li = r["half_adder"], r["tile"], r["lighting"]
    print("ring:", {k: round(v, 4) if isinstance(v, float) else v for k, v in r["ring"].items()})
    print("half adder levels (uA):")
    for row in ha["levels"]:
        print(f"  A={row['a']} B={row['b']}  carry {row['carry_uA']:8.2f} ±{row['carry_sigma_uA']:.2f}   sum {row['sum_uA']:8.3f} ±{row['sum_sigma_uA']:.2f}")
    d = ha["decisions"]
    print(f"  carry thr {d['carry']['threshold_uA']:.1f} Q {d['carry']['Q']:.1f} | sum thr {d['sum']['threshold_uA']:.1f} Q {d['sum']['Q']:.1f}")
    print(f"  XOR phase tolerance ±{ha['xor_phase_tolerance_deg']:.1f} deg, min laser {ha['min_laser_power_uW_for_Q7']:.1f} uW, "
          f"drift {ha['arm_phase_drift_deg_per_K_for_100um_arms']:.2f} deg/K, XOR extinction {ha['xor_extinction_dB']:.1f} dB")
    print("tile: g", round(tl["weight_scale_g"], 4), "T range", [round(v, 3) for v in tl["input_T_range"]])
    for k, v in tl["enob_by_rate"].items():
        print(f"  {k} GS/s ENOB:", " ".join(f"{e:.2f}" for e in v[::3]))
    print("  design ENOB", round(tl["design_enob"], 2), "budget", {k: round(v, 2) for k, v in tl["enob_with_one_impairment"].items()})
    print("lighting sigma_w", round(li["sigma_w"], 5), "validation", {k: round(v, 4) for k, v in li["validation"].items()})
    print("  sigma_y @ powers", [round(s, 5) for s in li["sigma_y"][::3]])
    print("  drift 0.1K", [round(v, 4) for v in li["drift_coeffs"][5]], " 0.5K", [round(v, 4) for v in li["drift_coeffs"][-1]])
