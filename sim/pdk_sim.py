#!/usr/bin/env python3
"""
Bench to Pixel: circuit simulation on published compact models (v2).

The on-chip circuits from the page, assembled from:
  * SiEPIC EBeam PDK compact models (via simphony 0.7.3): Y-branch, broadband
    directional coupler, half-ring coupler, strip waveguide, TE grating coupler.
  * Measured active devices from the literature:
      - depletion ring modulators: Karimelahi et al., "Optical and electrical trade-offs of
        rib-to-contact distance in depletion-type ring modulators", Opt. Express 25(17), 20202, 2017
      - Ge waveguide photodiode:   Li et al., Chin. Phys. Lett. 37(3), 038503, 2020
  * Free-carrier dispersion:      Soref & Bennett, IEEE J. Quantum Electron. 23(1), 1987
  * Receiver noise model:         Al-Qadasi et al., APL Photonics 7, 020902, 2022 (Eq. 8, Table 1)
  * Waveguide crossings:          Bogaerts et al., Opt. Lett. 32(19), 2801, 2007

The optical breadboard (the paper's bench) is not simulated here.

Validation (section 4) runs the same models against published numbers:
  V1  ring modulator: our model with Karimelahi's lifetimes vs their measured Q, ER, IL
      and loss-vs-voltage;
  V2  receiver + link budget: our implementation of Al-Qadasi's Eq. 8 and Eq. 13 vs their
      stated 85x85 binary MRR network limit.

Method notes
  * Half-adder cell: the whole cell is one SAX netlist of PDK S-matrices (grating coupler,
    Y-branches, broadband couplers, waveguides, Sagnac loop mirrors, MZI attenuators), solved
    as a scattering network so every back-reflection is included. Ring modulators enter as
    two-port elements from RingMod. Heaters (two MZI phases, the Michelson bias) are
    calibrated per chip. The same cell is re-solved at the PDK's thickness (210/220/230 nm)
    and width (480/500/520 nm) corners, and compared with a single-pass transfer-matrix
    composition to show how much the reflections matter.
  * 4x4 tile: single-pass transfer matrices (the PDK couplers reflect at -43 dB), Monte
    Carlo over random inputs and weights.
  * The raw SiEPIC half-ring S-matrix is slightly non-passive (largest singular value
    ~1.004 at 1550 nm). Inside a resonator that error compounds into gain, so the demux
    rings use the PDK's coupling strength |S14|^2 and the PDK waveguide's complex
    transmission in a passive (unitary-coupler) ring model instead of the raw matrix.
  * Assumptions (not from a PDK or a paper): 3 dB/cm strip waveguide loss, thermo-optic
    coefficient 1.86e-4 /K, 0.3% RMS weight-calibration residual, 6-bit input DACs, 8-bit
    heater DACs, and that the ADC samples at the end of each symbol.
  * No full-wave electromagnetic simulation is done beyond what the PDK data contains.

Run:  pip install -r requirements.txt && python3 pdk_sim.py      (about 1 minute)
"""
import json
import math
import os
import warnings

import numpy as np

warnings.filterwarnings("ignore")
import jax.numpy as jnp  # noqa: E402
import sax  # noqa: E402
from simphony.libraries import siepic  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
C0 = 299_792_458.0
QE = 1.602176634e-19
KB = 1.380649e-23
TK = 300.0
LAM0 = 1550e-9

SOURCES = {
    "pdk": "SiEPIC EBeam PDK compact models via simphony 0.7.3 (ebeam_y_1550, ebeam_bdc_te1550, ebeam_dc_halfring_straight, wg_integral_1550, ebeam_gc_te1550)",
    "ring_modulator": "Karimelahi et al., Optical and electrical trade-offs of rib-to-contact distance in depletion-type ring modulators, Optics Express 25(17):20202, 2017, doi:10.1364/OE.25.020202 (Table 1 amplitude decay times at -1 V; Sec. 4 dneff/dV of the doped section and d(1/tau_l)/dV; Fig. 6 ERmax/IL vs dV; Table 3 IL for 4 dB ER at 4 V and EO bandwidth)",
    "free_carrier": "Soref & Bennett, IEEE J. Quantum Electron. 23(1):123, 1987",
    "photodiode": "Li et al., Chinese Physics Letters 37(3):038503, 2020, doi:10.1088/0256-307X/37/3/038503",
    "receiver": "Al-Qadasi et al., APL Photonics 7:020902, 2022, arXiv:2109.08025v4 (Eq. 8, Eq. 13, Tables 1 and 3; N~85 for R = 1.2 A/W and a 10 dBm optical laser, Sec. III)",
    "crossing": "Bogaerts et al., Optics Letters 32(19):2801, 2007 (0.16 dB per crossing, crosstalk better than -40 dB)",
}

# ------------------------------------------------------------------ device parameters
RING_COMMON = dict(R=10e-6, ng=3.86, NA=5e17, ND=3e17, w_cm=500e-7, doped_frac=0.75)   # 75% of the circumference is doped
KARIMELAHI = {   # Table 1 amplitude decay times (-1 V), Sec. 4 slopes, Fig. 6 / Table 3 values kept for validation.
                 # f3dB_4V_GHz: Table 3 small-signal EO bandwidth at the detuning that keeps ER >= 4 dB for a 4 V swing
    "B": dict(gap_nm=250, tau_l=14.7e-12, tau_e=17.7e-12, dneff_per_V=1.88e-5,
              meas=dict(Q=4971, fwhm_pm=314.4, ER6=8.8, IL6=12.4, IL4dB4V=9.4, dinv=-4.4e8)),
    "C": dict(gap_nm=350, tau_l=71.2e-12, tau_e=59.1e-12, dneff_per_V=1.95e-5,
              meas=dict(Q=20068, fwhm_pm=77.4, ER6=17.8, IL6=3.8, IL4dB4V=2.5, dinv=-2.4e8, f3dB_4V_GHz=8.2)),
    "D": dict(gap_nm=400, tau_l=124e-12, tau_e=101.5e-12, dneff_per_V=2.38e-5,
              meas=dict(Q=34627, fwhm_pm=44.8, ER6=20.0, IL6=1.4, IL4dB4V=1.0, dinv=-2.6e8, f3dB_4V_GHz=5.8)),
}
PD = dict(resp=0.82, dark=5e-3 * 4e-4 * 8e-4)          # 0.82 A/W; 5 mA/cm^2 x (4 um x 8 um) = 1.6 nA
RX = dict(RL=50.0, RIN_dB=-140.0)                     # Al-Qadasi Table I
CROSS = dict(loss_dB=0.16, xtalk_dB=-40.0)           # Bogaerts 2007
WG_LOSS_DB_CM = 3.0                                    # strip waveguide loss used with the PDK model
DNDT_RIB = 1.8e-4                                       # thermo-optic dneff/dT, rib modulator ring
DNDT_STRIP = 1.86e-4                                    # thermo-optic dneff/dT, 500 x 220 nm strip
NG_STRIP = 4.2


def db2lin(x):
    return 10 ** (x / 10)


# ------------------------------------------------------------------ PDK component access
_SP_CACHE = {}


def sp(model, wl_m, **kw):
    """S-parameter dict from a simphony SiEPIC model, wavelength array in metres (memoised)."""
    wl = np.atleast_1d(np.asarray(wl_m, dtype=float))
    key = (getattr(model, "__name__", repr(model)), wl.tobytes(), tuple(sorted(kw.items())))
    if key not in _SP_CACHE:
        s = model(wl=wl * 1e6, **kw)
        _SP_CACHE[key] = {k: np.asarray(v).reshape(-1) for k, v in s.items()}
    return _SP_CACHE[key]


def wg_t(wl, length_um, loss=WG_LOSS_DB_CM):
    return sp(siepic.waveguide, wl, length=float(length_um), loss=loss)[("o0", "o1")]


def gc_t(wl):
    return sp(siepic.grating_coupler, wl)[("o0", "o1")]


def ybranch(wl):
    s = sp(siepic.y_branch, wl)
    return s


def bdc_T(wl):
    """2x2 transfer matrix of the broadband directional coupler: outputs (3,4) from inputs (1,2)."""
    s = sp(siepic.bidirectional_coupler, wl)
    return np.array([[s[("port_1", "port_3")], s[("port_2", "port_3")]],
                     [s[("port_1", "port_4")], s[("port_2", "port_4")]]])


def mzi_coupler(wl, theta, arm_um=50.0):
    """Thermo-optic MZI tunable coupler from two PDK BDCs. Returns (through, drop) field transmissions."""
    T1, T2 = bdc_T(wl), bdc_T(wl)
    ta = wg_t(wl, arm_um)
    arms = np.array([[ta, 0 * ta], [0 * ta, ta * np.exp(1j * theta)]])
    M = np.einsum("ijw,jkw,klw->ilw", T2, arms, T1)
    return M[0, 0], M[1, 0]


def loop_mirror(wl, loop_um=40.0):
    """Sagnac loop mirror: PDK Y-branch with its two arms joined by a PDK waveguide."""
    s = ybranch(wl)
    tl = wg_t(wl, loop_um)
    return s[("port_1", "port_2")] * tl * s[("port_3", "port_1")] + s[("port_1", "port_3")] * tl * s[("port_2", "port_1")] + s[("port_1", "port_1")]


def halfring_kappa2(wl, gap=150, radius=10):
    s = sp(siepic.half_ring, wl, gap=gap, radius=radius)
    return np.abs(s[("port_1", "port_4")]) ** 2


def add_drop(wl, lam_res, gap=150, radius=10):
    """Passive add-drop ring: PDK coupling strength, PDK waveguide round trip, heater-tuned to lam_res."""
    k2 = halfring_kappa2(wl, gap, radius)
    r = np.sqrt(1 - k2)
    L_um = 2 * np.pi * radius
    rt = wg_t(wl, L_um)                         # complex round trip from the PDK waveguide model
    rt0 = wg_t(np.array([lam_res]), L_um)[0]   # heater aligns resonance to lam_res
    rho = rt * np.exp(-1j * np.angle(rt0))
    half = np.sqrt(np.abs(rho)) * np.exp(1j * np.angle(rho) / 2)
    through = (r - r * rho) / (1 - r * r * rho)
    drop = -k2 * half / (1 - r * r * rho)
    return through, drop


# ------------------------------------------------------------------ ring modulator (validated in V1)
NI = 1.0e10
EPS_SI = 11.7 * 8.8541878128e-14
VBI = KB * TK / QE * math.log(RING_COMMON["NA"] * RING_COMMON["ND"] / NI ** 2)
DN_E = 8.8e-22 * RING_COMMON["ND"]
DN_H = 8.5e-18 * RING_COMMON["NA"] ** 0.8
DA_E = 8.5e-18 * RING_COMMON["ND"]
DA_H = 6.0e-18 * RING_COMMON["NA"]


def _wnp(V):
    NA, ND = RING_COMMON["NA"], RING_COMMON["ND"]
    W = math.sqrt(2 * EPS_SI * (VBI + V) / QE * (NA + ND) / (NA * ND))
    return W * NA / (NA + ND), W * ND / (NA + ND)


def _raw(V):
    """Index and absorption change per unit overlap, relative to -1 V (reverse bias V > 0)."""
    wn, wp = _wnp(V)
    wn1, wp1 = _wnp(1.0)
    w = RING_COMMON["w_cm"]
    return ((wn - wn1) * DN_E + (wp - wp1) * DN_H) / w, -((wn - wn1) * DA_E + (wp - wp1) * DA_H) / w


class RingMod:
    """All-pass depletion ring from Karimelahi et al. Their Table 1 time constants are field-amplitude
    decay times (Q = omega tau / 2), giving round-trip loss and coupling at -1 V; depletion-width changes
    with Soref-Bennett coefficients give index and loss vs bias. One overlap factor is calibrated to the
    paper's d(neff)/dV of the doped section at -1 V; both changes are then averaged over the ring."""

    def __init__(self, name):
        p = KARIMELAHI[name]
        self.name, self.p = name, p
        self.L = 2 * math.pi * RING_COMMON["R"]
        self.ng = RING_COMMON["ng"]
        self.trt = self.ng * self.L / C0
        h = 1e-4
        self.G = p["dneff_per_V"] / ((_raw(1 + h)[0] - _raw(1 - h)[0]) / (2 * h))
        self.r = math.exp(-self.trt / p["tau_e"])

    def dn(self, V):
        """Ring-averaged index change. Karimelahi's dneff/dV is for the doped section, so it is scaled
        by the doped fraction of the circumference (their Delta n_eff = n_g Delta lambda / (F lambda))."""
        return RING_COMMON["doped_frac"] * self.G * np.vectorize(lambda v: _raw(v)[0])(V)

    def inv_tau_l(self, V):
        da = RING_COMMON["doped_frac"] * self.G * np.vectorize(lambda v: _raw(v)[1])(V)   # 1/cm, power, ring-averaged
        return 1 / self.p["tau_l"] + 0.5 * (C0 * 100 / self.ng) * da

    def t(self, wl, V, lam_res_1V, dT=0.0):
        """Field transmission. lam_res_1V: resonance at -1 V and design temperature."""
        V = np.asarray(V, dtype=float)
        a = np.exp(-self.trt * self.inv_tau_l(V))
        lres = lam_res_1V + LAM0 * self.dn(V) / self.ng + LAM0 * DNDT_RIB * dT / self.ng
        phi = 2 * np.pi * self.ng * self.L * (lres - wl) / LAM0 ** 2
        e = a * np.exp(1j * phi)
        return (self.r - e) / (1 - self.r * e)

    def fwhm(self):
        return 2 * (1 / self.p["tau_l"] + 1 / self.p["tau_e"]) / (2 * math.pi) * LAM0 ** 2 / C0

    def predicted_dinv_per_V(self):
        h = 1e-4
        return (self.inv_tau_l(1 + h) - self.inv_tau_l(1 - h)) / (2 * h)


DT_PM_PER_K = LAM0 * DNDT_RIB / RING_COMMON["ng"] * 1e12


# ------------------------------------------------------------------ receiver noise (Al-Qadasi Eq. 8 form)
def rx_sigma(i_mean, rate, resp=PD["resp"], dark=PD["dark"], RL=RX["RL"], RIN_dB=RX["RIN_dB"]):
    """Current noise std: shot (signal+dark), load thermal, laser RIN; bandwidth DR/sqrt(2)."""
    be = rate / math.sqrt(2)
    i_sig = np.asarray(i_mean) - dark
    var = 2 * QE * np.asarray(i_mean) + 4 * KB * TK / RL + (i_sig ** 2) * db2lin(RIN_dB)
    return np.sqrt(var * be)


def q_ber(qf):
    return 0.5 * math.erfc(qf / math.sqrt(2))


# ================================================================== 1. half-adder cell
HA = dict(ring="D", v0=0.0, v1=4.0, rate=1e9, p_fiber=1e-3,
          L_branch_um=165.0, L_s1s2_um=70.0, L_s2s3_um=80.0, L_armA_um=87.5, L_armB_um=52.5, L_out_um=120.0,
          mzi_arm_um=50.0, loop_um=40.0)
NOMINAL = (220.0, 500.0, 0.0)          # silicon thickness (nm), waveguide width (nm), grating tooth width offset (nm)
# on-chip corners: thickness and width vary every PDK component; the grating coupler's tooth width is
# tabulated separately (gc_corners) because it moves the coupling peak by ~50 nm and acts as a fibre-coupling loss
CORNERS = [(t, w, 0.0) for t in (210.0, 220.0, 230.0) for w in (480.0, 500.0, 520.0)]


def _elem(wl=1.55, re=1.0, im=0.0):
    """Two-port element with a given complex transmission (used for the ring modulators)."""
    wl = jnp.asarray(wl)
    return sax.reciprocal({("o0", "o1"): (re + 1j * im) * jnp.ones_like(wl)})


def _phase(wl=1.55, phi=0.0):
    """Heater phase shifter."""
    wl = jnp.asarray(wl)
    return sax.reciprocal({("o0", "o1"): jnp.exp(1j * phi) * jnp.ones_like(wl)})


SAX_MODELS = {"bdc": siepic.bidirectional_coupler, "wg": siepic.waveguide, "yb": siepic.y_branch,
              "gc": siepic.grating_coupler, "el": _elem, "ps": _phase}


def ha_netlist(th=220.0, w=500.0, gdw=0.0):
    """The cell from the page as a SAX netlist. Fibre -> grating coupler -> S1, S2 (Y-branches) -> S3 (broadband
    coupler) Michelson with ring modulators and Sagnac loop mirrors in both arms; carry branches with a ring
    modulator and an MZI attenuator each, ending on the two inputs of one photodiode."""
    def wg(L):
        return {"component": "wg", "settings": {"length": float(L), "loss": WG_LOSS_DB_CM, "width": w, "height": th}}
    yb = {"component": "yb", "settings": {"thickness": th, "width": w}}
    bdc = {"component": "bdc", "settings": {"thickness": th, "width": w}}
    ins = {"gc": {"component": "gc", "settings": {"thickness": th, "dwidth": gdw}}, "s1": yb, "s2": yb, "s3": bdc,
           "w12": wg(HA["L_s1s2_um"]), "w23": wg(HA["L_s2s3_um"]), "wout": wg(HA["L_out_um"])}
    con = {"gc,o1": "s1,port_1", "s1,port_3": "w12,o0", "w12,o1": "s2,port_1", "s2,port_3": "w23,o0",
           "w23,o1": "s3,port_1", "s3,port_2": "wout,o0"}
    ports = {"fibre": "gc,o0", "sum": "wout,o1"}
    for X, src in (("A", "s1,port_2"), ("B", "s2,port_2")):
        ins.update({f"c{X}": "el", f"m{X}1": bdc, f"m{X}2": bdc, f"m{X}a": wg(HA["mzi_arm_um"]),
                    f"m{X}b": wg(HA["mzi_arm_um"]), f"m{X}p": "ps", f"o{X}": wg(HA["L_branch_um"])})
        con.update({src: f"c{X},o0", f"c{X},o1": f"m{X}1,port_1", f"m{X}1,port_3": f"m{X}a,o0",
                    f"m{X}a,o1": f"m{X}2,port_1", f"m{X}1,port_4": f"m{X}p,o0", f"m{X}p,o1": f"m{X}b,o0",
                    f"m{X}b,o1": f"m{X}2,port_2", f"m{X}2,port_3": f"o{X},o0"})
        ports.update({f"carry{X}": f"o{X},o1", f"mzi{X}_in2": f"m{X}1,port_2", f"mzi{X}_drop": f"m{X}2,port_4"})
    for X, L, port in (("A", HA["L_armA_um"], "s3,port_3"), ("B", HA["L_armB_um"], "s3,port_4")):
        ins.update({f"a{X}": wg(L), f"q{X}": "ps", f"r{X}": "el", f"y{X}": yb, f"l{X}": wg(HA["loop_um"])})
        con.update({port: f"a{X},o0", f"a{X},o1": f"q{X},o0", f"q{X},o1": f"r{X},o0", f"r{X},o1": f"y{X},port_1",
                    f"y{X},port_2": f"l{X},o0", f"l{X},o1": f"y{X},port_3"})
    return {"instances": ins, "connections": con, "ports": ports}


BITS_A, BITS_B = np.array([0, 0, 1, 1]), np.array([0, 1, 0, 1])


class HalfAdder:
    """The whole cell solved as one scattering network in SAX from the PDK S-matrices (all back-reflections
    included). Ring modulators enter as two-port elements from RingMod. Heaters (two MZI phases and the
    Michelson bias) are calibrated per chip, as they would be after fabrication."""

    def __init__(self, corner=NOMINAL):
        self.corner = corner
        self.ring = RingMod(HA["ring"])
        lam = LAM0 + np.linspace(-150e-12, 150e-12, 6001)
        t0 = np.abs(self.ring.t(lam, HA["v0"], LAM0)) ** 2
        t1 = np.abs(self.ring.t(lam, HA["v1"], LAM0)) ** 2
        self.lam = float(lam[int(np.argmax(t1 / t0))])      # laser line: largest 0 V / 4 V contrast
        wl = np.array([self.lam])
        self.tr = {0: complex(self.ring.t(wl, HA["v0"], LAM0)[0]), 1: complex(self.ring.t(wl, HA["v1"], LAM0)[0])}
        self.circ, _ = sax.circuit(netlist=ha_netlist(*corner), models=SAX_MODELS)

    def run(self, a, b, ta, tb, phi):
        """Carry and sum powers per watt in the fibre. All arguments broadcast (vectorised over configurations)."""
        a, b, ta, tb, phi = np.broadcast_arrays(*(np.atleast_1d(np.asarray(v, dtype=float)) for v in (a, b, ta, tb, phi)))
        shp = a.shape
        tA = np.where(a.ravel() > 0.5, self.tr[1], self.tr[0])
        tB = np.where(b.ravel() > 0.5, self.tr[1], self.tr[0])
        el = lambda t: {"re": jnp.asarray(t.real), "im": jnp.asarray(t.imag)}  # noqa: E731
        s = self.circ(wl=jnp.full(a.size, self.lam * 1e6), cA=el(tA), cB=el(tB), rA=el(tA), rB=el(tB),
                      mAp={"phi": jnp.asarray(ta.ravel())}, mBp={"phi": jnp.asarray(tb.ravel())},
                      qB={"phi": jnp.asarray(phi.ravel() / 2)})
        g = lambda k: np.abs(np.asarray(s["fibre", k])) ** 2  # noqa: E731
        return (g("carryA") + g("carryB")).reshape(shp), g("sum").reshape(shp)

    @staticmethod
    def _refine(f, x0, span, n=401):
        xs = x0 + np.linspace(-span, span, n)
        return float(xs[int(np.argmin(f(xs)))])

    def calibrate(self):
        th = np.linspace(0, 2 * np.pi, 721, endpoint=False)
        step = th[1]
        tb = float(th[int(np.argmax(self.run(0, 1, 0.0, th, 0.0)[0]))])
        tb = self._refine(lambda x: -self.run(0, 1, 0.0, x, 0.0)[0], tb, step)
        ta = tb
        for _ in range(2):   # carry marking: P(1,0) = (2/3) P(0,1), the polarisers' job on the bench
            target = (2 / 3) * self.run(0, 1, ta, tb, 0.0)[0][0]
            c10 = self.run(1, 0, th, tb, 0.0)[0]
            i_lo, i_hi = int(np.argmin(c10)), int(np.argmax(c10))
            idx = np.arange(min(i_lo, i_hi), max(i_lo, i_hi) + 1)
            ta = float(th[idx[int(np.argmin(np.abs(c10[idx] - target)))]])
            ta = self._refine(lambda x: np.abs(self.run(1, 0, x, tb, 0.0)[0] - target), ta, step)
        ph = np.linspace(-np.pi, np.pi, 1441)
        p0 = float(ph[int(np.argmin(self.run(1, 1, ta, tb, ph)[1]))])
        phi = self._refine(lambda x: self.run(1, 1, ta, tb, x)[1], p0, ph[1] - ph[0])
        self.ta, self.tb, self.phi = ta, tb, phi
        self.unit = self.run(BITS_A, BITS_B, ta, tb, phi)
        return self

    def levels(self, p_fiber=None, unit=None, rate=None):
        p_fiber = HA["p_fiber"] if p_fiber is None else p_fiber
        rate = rate or HA["rate"]
        c, s = self.unit if unit is None else unit
        rows = []
        for k in range(4):
            ic = PD["resp"] * p_fiber * c[k] + PD["dark"]
            isum = PD["resp"] * p_fiber * s[k] + PD["dark"]
            rows.append(dict(a=int(BITS_A[k]), b=int(BITS_B[k]), carry_uA=ic * 1e6, sum_uA=isum * 1e6,
                             carry_sigma_uA=float(rx_sigma(ic, rate)) * 1e6, sum_sigma_uA=float(rx_sigma(isum, rate)) * 1e6))
        return rows

    def min_power(self, q=7.0):
        lo, hi = 1e-6, 1e-1
        for _ in range(60):
            mid = math.sqrt(lo * hi)
            lo, hi = (lo, mid) if decide(self.levels(mid))["min_Q"] >= q else (mid, hi)
        return hi


def ha_single_pass(hc, bits, ta, tb, phi):
    """The same cell composed with single-pass transfer matrices (no back-reflections), for comparison."""
    wl = np.array([hc.lam])
    yb = ybranch(wl)
    s12, s13 = yb[("port_1", "port_2")], yb[("port_1", "port_3")]
    e_in = gc_t(wl)
    tr = lambda b: hc.tr[b]  # noqa: E731
    a_bit, b_bit = bits
    eA = e_in * s12
    e1 = e_in * s13 * wg_t(wl, HA["L_s1s2_um"])
    eB = e1 * s12
    e3 = e1 * s13 * wg_t(wl, HA["L_s2s3_um"])
    cA = eA * tr(a_bit) * mzi_coupler(wl, ta, HA["mzi_arm_um"])[0] * wg_t(wl, HA["L_branch_um"])
    cB = eB * tr(b_bit) * mzi_coupler(wl, tb, HA["mzi_arm_um"])[0] * wg_t(wl, HA["L_branch_um"])
    T = bdc_T(wl)
    lm = loop_mirror(wl, HA["loop_um"])
    rhoA = wg_t(wl, HA["L_armA_um"]) ** 2 * tr(a_bit) ** 2 * lm
    rhoB = wg_t(wl, HA["L_armB_um"]) ** 2 * tr(b_bit) ** 2 * lm * np.exp(1j * phi)
    eS = e3 * (T[0, 0] * rhoA * T[0, 1] + T[1, 0] * rhoB * T[1, 1]) * wg_t(wl, HA["L_out_um"])
    return float(np.abs(cA[0]) ** 2 + np.abs(cB[0]) ** 2), float(np.abs(eS[0]) ** 2)


def decide(rows):
    out = {}
    for key, fn in (("carry", lambda a, b: a & b), ("sum", lambda a, b: a ^ b)):
        ones = [(r[key + "_uA"], r[key + "_sigma_uA"]) for r in rows if fn(r["a"], r["b"]) == 1]
        zeros = [(r[key + "_uA"], r[key + "_sigma_uA"]) for r in rows if fn(r["a"], r["b"]) == 0]
        hi0, lo1 = max(zeros), min(ones)
        thr = (hi0[0] + lo1[0]) / 2
        qf = (lo1[0] - hi0[0]) / (lo1[1] + hi0[1])
        out[key] = dict(threshold_uA=thr, Q=qf, BER=q_ber(qf))
    out["min_Q"] = min(out["carry"]["Q"], out["sum"]["Q"])
    return out


def half_adder_report():
    hc = HalfAdder().calibrate()
    rows = hc.levels()
    dec = decide(rows)
    # XOR bias error: sum output for one arm open (1,0) and both open (1,1), threshold fixed at its calibrated value
    phases = np.linspace(-90, 90, 61)
    _, s10 = hc.run(1, 0, hc.ta, hc.tb, hc.phi + np.deg2rad(phases))
    _, s11 = hc.run(1, 1, hc.ta, hc.tb, hc.phi + np.deg2rad(phases))
    i10 = PD["resp"] * HA["p_fiber"] * s10 + PD["dark"]
    i11 = PD["resp"] * HA["p_fiber"] * s11 + PD["dark"]
    thr = dec["sum"]["threshold_uA"] * 1e-6
    margin = np.minimum((thr - i11) / rx_sigma(i11, HA["rate"]), (i10 - thr) / rx_sigma(i10, HA["rate"]))
    ok = phases[(phases >= 0)][np.cumprod(margin[phases >= 0] >= 7.0).astype(bool)]
    tol = float(ok.max()) if ok.size else 0.0
    # single-pass transfer-matrix composition vs the full scattering solution, same heater settings
    sp_vs_full = []
    for k in range(4):
        c_sp, s_sp = ha_single_pass(hc, (int(BITS_A[k]), int(BITS_B[k])), hc.ta, hc.tb, hc.phi)
        sp_vs_full.append(dict(a=int(BITS_A[k]), b=int(BITS_B[k]), carry_ratio=float(hc.unit[0][k] / c_sp),
                               sum_ratio=float(hc.unit[1][k] / s_sp)))
    # process corners from the PDK (silicon thickness, waveguide width, grating width): recalibrated vs nominal heaters
    corners = []
    for cn in CORNERS:
        hk = HalfAdder(cn)
        fixed = hk.run(BITS_A, BITS_B, hc.ta, hc.tb, hc.phi)
        hk.calibrate()
        d = decide(hk.levels())
        corners.append(dict(thickness_nm=cn[0], width_nm=cn[1], gc_dwidth_nm=cn[2],
                            carry_Q=d["carry"]["Q"], sum_Q=d["sum"]["Q"], min_Q=d["min_Q"],
                            min_fibre_power_uW=hk.min_power() * 1e6,
                            min_Q_with_nominal_heaters=decide(hk.levels(unit=fixed))["min_Q"]))
    wl = np.array([hc.lam])
    wls = np.linspace(1.50e-6, 1.60e-6, 2001)
    gc_corners = []
    for th in (210.0, 220.0, 230.0):
        for dw in (-20.0, 0.0, 20.0):
            g = np.abs(sp(siepic.grating_coupler, wls, thickness=th, dwidth=dw)[("o0", "o1")]) ** 2
            g0 = np.abs(sp(siepic.grating_coupler, wl, thickness=th, dwidth=dw)[("o0", "o1")][0]) ** 2
            gc_corners.append(dict(thickness_nm=th, dwidth_nm=dw, loss_at_laser_dB=float(-10 * np.log10(g0)),
                                   peak_nm=float(wls[int(np.argmax(g))] * 1e9), peak_loss_dB=float(-10 * np.log10(g.max()))))
    yb = ybranch(wl)
    budget = {
        "grating_coupler_dB": float(10 * np.log10(np.abs(gc_t(wl)[0]) ** 2)),
        "y_branch_each_output_dB": float(10 * np.log10(np.abs(yb[("port_1", "port_2")][0]) ** 2)),
        "loop_mirror_reflection_single_pass_dB": float(10 * np.log10(np.abs(loop_mirror(wl, HA["loop_um"])[0]) ** 2)),
        "ring_bit1_through_dB": float(10 * np.log10(abs(hc.tr[1]) ** 2)),
        "ring_bit0_through_dB": float(10 * np.log10(abs(hc.tr[0]) ** 2)),
    }
    k_one = math.degrees(2 * math.pi * DNDT_STRIP * 2 * HA["L_armA_um"] * 1e-6 / LAM0)
    k_uni = math.degrees(2 * math.pi * DNDT_STRIP * 2 * abs(HA["L_armA_um"] - HA["L_armB_um"]) * 1e-6 / LAM0)
    f3 = KARIMELAHI[HA["ring"]]["meas"]["f3dB_4V_GHz"] * 1e9
    return {
        "solver": "SAX scattering solution of the full cell from SiEPIC EBeam S-matrices",
        "laser_mW_in_fibre": HA["p_fiber"] * 1e3, "rate_Gbps": HA["rate"] / 1e9, "ring": "Karimelahi Ring D",
        "drive_V": [HA["v0"], HA["v1"]], "laser_detuning_pm": (hc.lam - LAM0) * 1e12,
        "heaters_rad": dict(mzi_A=hc.ta, mzi_B=hc.tb, michelson_bias=hc.phi),
        "levels": rows, "decisions": dec, "budget": budget,
        "xor_phase_sweep": dict(phase_deg=phases.tolist(), both_open_uA=(i11 * 1e6).tolist(),
                                one_open_uA=(i10 * 1e6).tolist(), threshold_uA=thr * 1e6),
        "xor_phase_tolerance_deg": tol, "min_fibre_power_uW_for_Q7": hc.min_power() * 1e6,
        "xor_deg_per_K_one_arm_heated": k_one, "xor_deg_per_K_uniform": k_uni,
        "xor_tolerance_K_one_arm_heated": tol / k_one, "xor_tolerance_K_uniform": tol / k_uni,
        "isi_residual_end_of_bit": math.exp(-2 * math.pi * f3 / HA["rate"]),
        "single_pass_vs_full": sp_vs_full, "corners": corners, "gc_corners": gc_corners,
    }


# ================================================================== 2. 4x4 WDM tile
TILE = dict(ring="C", vmax=4.0, spacing_nm=1.6, n=4, pitch_um=50.0, dac_bits=6, heater_bits=8, cal_sigma=0.003)
LAMS = LAM0 + np.arange(4) * TILE["spacing_nm"] * 1e-9


class Tile:
    def __init__(self):
        self.ring = RingMod(TILE["ring"])
        # modulator window: ring resonance offset that maximises the 0..4 V transmission span, monotonic
        best = (-1, 0.0)
        V = np.linspace(0, TILE["vmax"], 81)
        for off in np.linspace(-150e-12, 150e-12, 601):
            T = np.abs(self.ring.t(LAM0, V, LAM0 + off)) ** 2
            if np.all(np.diff(T) > 0) and T[-1] - T[0] > best[0]:
                best = (T[-1] - T[0], off)
        self.off = best[1]
        self.Vg = np.linspace(0, TILE["vmax"], 4001)
        self.Tg = np.abs(self.ring.t(LAM0, self.Vg, LAM0 + self.off)) ** 2
        self.T_lo, self.T_hi = float(self.Tg[0]), float(self.Tg[-1])
        # demultiplexer: cascaded add-drop rings, one per row; power matrix X[row j, line k]
        X = np.zeros((4, 4))
        bus = np.ones(4)
        for j in range(4):
            th, dr = add_drop(LAMS, LAMS[j])
            X[j] = bus * np.abs(dr) ** 2
            bus = bus * np.abs(th) ** 2
        self.X = X
        self.gc = np.abs(gc_t(LAMS)) ** 2
        # tunable-coupler weights: drop(theta) and through(theta) at each row wavelength
        th = np.linspace(0, 2 * np.pi, 2 ** TILE["heater_bits"], endpoint=False)
        self.th = th
        self.cpl = [mzi_coupler(np.full(th.shape, l), th) for l in LAMS]
        self.d_min = min(float(np.min(np.abs(c[1]) ** 2)) for c in self.cpl)
        self.d_max = min(float(np.max(np.abs(c[1]) ** 2)) for c in self.cpl)
        self.t_cross = db2lin(-CROSS["loss_dB"])
        self.wgp = float(np.abs(wg_t(np.array([LAM0]), TILE["pitch_um"])[0]) ** 2)
        self.G = self._gain()

    # modulator: x in [0,1] -> transmission, with 6-bit DAC predistortion, ring drift
    def encode(self, x, dT=0.0, dac=True):
        target = self.T_lo + np.clip(x, 0, 1) * (self.T_hi - self.T_lo)
        V = np.interp(target, self.Tg, self.Vg)
        if dac:
            lv = 2 ** TILE["dac_bits"] - 1
            V = np.round(V / TILE["vmax"] * lv) / lv * TILE["vmax"]
        if dT == 0.0:
            return np.interp(V, self.Vg, self.Tg)
        return np.abs(self.ring.t(LAM0, V, LAM0 + self.off, dT=dT)) ** 2

    def demux_gain(self, dT=0.0):
        """Row power of each row's own line relative to the calibrated (dT=0) value, uniform drift."""
        if dT == 0.0:
            return 1.0
        shift = (LAM0 * DNDT_STRIP / NG_STRIP) * dT        # strip demux ring
        th, dr = add_drop(LAMS[:1] - shift, LAMS[0])
        th0, dr0 = add_drop(LAMS[:1], LAMS[0])
        return float(np.abs(dr[0]) ** 2 / np.abs(dr0[0]) ** 2)

    def path_loss(self, i, j):
        return self.t_cross ** (3 - j) * (self.wgp ** (i + 1 + (3 - j)))

    def _tab(self, j):
        d, t = np.abs(self.cpl[j][1]) ** 2, np.abs(self.cpl[j][0]) ** 2
        o = np.argsort(d)
        return d[o], t[o]

    def thru(self, j, d):
        """Through-port power of input j's coupler set to drop d (PDK MZI, includes its excess loss)."""
        dd, tt = self._tab(j)
        return np.interp(d, dd, tt)

    def _gain(self):
        def ok(g):
            for j in range(4):
                rem = 1.0
                for i in range(4):
                    d = g / (rem * self.path_loss(i, j))
                    if d > self.d_max:
                        return False
                    rem *= self.thru(j, d)
            return True
        lo, hi = 0.0, 1.0
        for _ in range(60):
            m = (lo + hi) / 2
            lo, hi = (m, hi) if ok(m) else (lo, m)
        return lo

    def realise(self, w, rng, quant=True, cal=True):
        """Target weights [..., i, j] -> realised effective transmissions / G, through the PDK coupler."""
        w = np.asarray(w, dtype=float)
        out = np.empty_like(w)
        for j in range(w.shape[-1]):
            drops, thrus = self._tab(j)
            order = np.arange(len(drops))
            rem = np.ones(w.shape[:-2])
            for i in range(w.shape[-2]):
                loss = self.path_loss(i, j)
                d_t = np.clip(self.G * w[..., i, j] / (rem * loss), self.d_min, self.d_max)
                if quant:
                    k = np.clip(np.searchsorted(drops[order], d_t), 1, len(order) - 1)
                    lo_, hi_ = order[k - 1], order[k]
                    pick = np.where(np.abs(drops[lo_] - d_t) < np.abs(drops[hi_] - d_t), lo_, hi_)
                    d, t_ = drops[pick], thrus[pick]
                else:
                    d, t_ = d_t, self.thru(j, d_t)
                out[..., i, j] = d * rem * loss / self.G
                rem = rem * t_
        if cal:
            out = out + rng.normal(0, TILE["cal_sigma"], out.shape)
        return out


def tile_mc(tile, p_line, rate, n=10000, adc_bits=10, dT=0.0, rng=None, impair=("dac", "weights", "noise", "adc")):
    rng = rng or np.random.default_rng(1)
    x = rng.random((n, 4))
    w = rng.random((n, 4, 4))
    y_true = np.einsum("nij,nj->ni", w, x) / 4
    t = tile.encode(x, dT, dac="dac" in impair)
    w_r = tile.realise(w, rng, quant="weights" in impair, cal="weights" in impair)
    p_row = p_line * tile.gc[None, :] * np.diag(tile.X)[None, :] * tile.demux_gain(dT)    # own line per row
    leak = p_line * (tile.gc[None, :] * tile.X).sum(1) - p_line * tile.gc * np.diag(tile.X)  # other lines, unmodulated
    p_ij = (p_row * t)[:, None, :] * tile.G * w_r + leak[None, None, :] * tile.G * w_r
    i_mean = PD["resp"] * p_ij.sum(-1) + PD["dark"]
    # crossing crosstalk: each column picks up -40 dB of every row it crosses (static, calibrated out)
    if "noise" in impair:
        i_meas = i_mean + rx_sigma(i_mean, rate) * rng.standard_normal(i_mean.shape)
    else:
        i_meas = i_mean
    p0 = p_line * tile.gc * np.diag(tile.X)
    offset = PD["resp"] * ((p0 * tile.T_lo + leak) [None, None, :] * tile.G * w).sum(-1) + PD["dark"]
    scale = PD["resp"] * (p0 * (tile.T_hi - tile.T_lo))[None, None, :] * tile.G
    y = (i_meas - offset) / (scale.mean(-1) * 4)
    if "adc" in impair:
        lv = 2 ** adc_bits - 1
        y = np.round(np.clip(y, 0, 1) * lv) / lv
    s = float(np.std(y - y_true))
    return s, -math.log2(math.sqrt(12) * s)


def tile_report(tile):
    rng = np.random.default_rng(3)
    powers = np.logspace(-2, math.log10(10), 22)    # mW per comb line in fibre
    sweep = {}
    for rate in (1e9, 2.5e9, 5e9):
        sweep[f"{rate / 1e9:g}"] = [tile_mc(tile, p * 1e-3, rate, rng=rng)[1] for p in powers]
    full = tile_mc(tile, 2e-3, 5e9, adc_bits=8, rng=rng)
    budget = {k: tile_mc(tile, 2e-3, 5e9, adc_bits=8, rng=rng, impair=(k,))[1] for k in ("noise", "dac", "weights", "adc")}
    xt = tile.X / np.diag(tile.X)[:, None]
    return {
        "channels": 4, "channel_spacing_nm": TILE["spacing_nm"], "modulator": "Karimelahi Ring C, 0-4 V",
        "modulator_T_range": [tile.T_lo, tile.T_hi], "weight_scale_g": tile.G,
        "coupler_drop_range": [tile.d_min, tile.d_max],
        "demux_worst_crosstalk_dB": float(10 * np.log10(np.max(xt[~np.eye(4, dtype=bool)]))),
        "grating_coupler_dB": [float(10 * np.log10(v)) for v in tile.gc],
        "power_mW": powers.tolist(), "enob_by_rate": sweep,
        "isi_residual_end_of_symbol": {f"{r:g}": math.exp(-2 * math.pi * KARIMELAHI["C"]["meas"]["f3dB_4V_GHz"] * 1e9 / (r * 1e9)) for r in (1, 2.5, 5)},
        "demux_thermal_shift_pm_per_K": LAM0 * DNDT_STRIP / NG_STRIP * 1e12,
        "design_point": dict(p_mW=2.0, rate_GSps=5, adc_bits=8), "design_enob": full[1], "design_rms_error": full[0],
        "enob_with_one_impairment": budget,
    }


# ================================================================== 3. lighting N.L on the tile
SQ3 = math.sqrt(3)


def unit_vectors(rng, n):
    v = rng.normal(size=(n, 3))
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def lighting_circuit(tile, p_line, rate, n=12000, adc_bits=8, dT=0.0, rng=None, impair=("dac", "weights", "noise", "adc")):
    rng = rng or np.random.default_rng(5)
    nrm, lgt = unit_vectors(rng, n), unit_vectors(rng, n)
    x = np.concatenate([(nrm + 1) / 2, np.zeros((n, 1))], 1)
    lp, lm = np.maximum(lgt, 0), np.maximum(-lgt, 0)
    w = np.zeros((n, 4, 4))
    w[:, 0, :3], w[:, 1, :3] = lp, lm
    t = tile.encode(x, dT, dac="dac" in impair)
    w_r = tile.realise(w, rng, quant="weights" in impair, cal="weights" in impair)
    p_row = p_line * tile.gc * np.diag(tile.X) * tile.demux_gain(dT)
    leak = p_line * (tile.gc[None, :] * tile.X).sum(1) - p_line * tile.gc * np.diag(tile.X)
    p_ij = (p_row * t)[:, None, :] * tile.G * w_r + leak[None, None, :] * tile.G * w_r
    i_mean = PD["resp"] * p_ij.sum(-1)[:, :2] + PD["dark"]
    i_meas = i_mean + (rx_sigma(i_mean, rate) * rng.standard_normal(i_mean.shape) if "noise" in impair else 0)
    p0 = p_line * tile.gc * np.diag(tile.X)
    offset = PD["resp"] * ((p0 * tile.T_lo + leak)[None, None, :] * tile.G * w).sum(-1)[:, :2] + PD["dark"]
    scale = PD["resp"] * float(np.mean(p0 * (tile.T_hi - tile.T_lo))) * tile.G
    y = ((i_meas[:, 0] - offset[:, 0]) - (i_meas[:, 1] - offset[:, 1])) / scale
    if "adc" in impair:
        lv = 2 ** adc_bits - 1
        y = np.round((np.clip(y, -SQ3, SQ3) + SQ3) / (2 * SQ3) * lv) / lv * 2 * SQ3 - SQ3
    nl = 2 * y - lgt.sum(1)
    return float(np.std(nl - (nrm * lgt).sum(1)))


def drift_poly(tile, dT):
    x = np.linspace(0, 1, 201)
    xp = (tile.encode(x, dT, dac=False) * tile.demux_gain(dT) - tile.T_lo) / (tile.T_hi - tile.T_lo)
    c2, c1, c0 = np.polyfit(x, xp - x, 2)
    return [float(c0), float(c1), float(c2)]


def browser_model(sig_y, coeffs, adc_bits, rng, sigma_w, n=20000, dac_bits=6):
    nrm, lgt = unit_vectors(rng, n), unit_vectors(rng, n)
    x = (nrm + 1) / 2
    lv_in = 2 ** dac_bits - 1
    x = np.round(x * lv_in) / lv_in
    c0, c1, c2 = coeffs
    x = x + c0 + c1 * x + c2 * x * x
    lw = lgt + rng.normal(0, sigma_w, lgt.shape)
    y = (lw * x).sum(1) + sig_y * rng.standard_normal(n)
    lv = 2 ** adc_bits - 1
    y = np.round((np.clip(y, -SQ3, SQ3) + SQ3) / (2 * SQ3) * lv) / lv * 2 * SQ3 - SQ3
    nl = 2 * y - lgt.sum(1)
    return float(np.std(nl - (nrm * lgt).sum(1)))


def lighting_report(tile):
    rng = np.random.default_rng(11)
    rate = 5e9
    powers = np.logspace(-2, math.log10(10), 22)
    sig_y = [lighting_circuit(tile, p * 1e-3, rate, rng=rng, impair=("noise",)) / 2 for p in powers]
    temps = np.linspace(0, 0.3, 31)
    drift = [drift_poly(tile, t) for t in temps]
    w = rng.random((3000, 4, 4))
    w[:, 2:, :] = 0
    sigma_w = float(np.std((tile.realise(w, rng) - w)[:, :2, :3]))
    sy = lambda p: float(10 ** np.interp(math.log10(p), np.log10(powers), np.log10(sig_y)))  # noqa: E731
    val = {}
    for p, dT in ((2.0, 0.0), (0.5, 0.0), (2.0, 0.05)):
        val[f"{p}mW_{dT}K"] = dict(circuit=lighting_circuit(tile, p * 1e-3, rate, rng=rng, dT=dT),
                                   page=browser_model(sy(p), drift_poly(tile, dT), 8, rng, sigma_w))
    return {
        "rate_GSps": 5, "power_mW": powers.tolist(), "sigma_y": sig_y,
        "drift_dT_K": temps.tolist(), "drift_coeffs": drift, "sigma_w": sigma_w, "dac_bits": TILE["dac_bits"],
        "default": {"p_mW": 2.0, "adc_bits": 8, "dT_K": 0.0}, "validation": val,
    }


# ================================================================== 4. validation against published results
def validate_rings():
    out = {}
    lam = LAM0 + np.linspace(-500e-12, 500e-12, 50001)
    for name in ("B", "C", "D"):
        rm = RingMod(name)
        m = KARIMELAHI[name]["meas"]
        T0 = np.abs(rm.t(lam, 0.0, LAM0)) ** 2
        res = {}
        for dV in (4.0, 6.0):
            T1 = np.abs(rm.t(lam, dV, LAM0)) ** 2
            er = 10 * np.log10(np.maximum(T1, T0) / np.minimum(T1, T0))
            il = -10 * np.log10(np.maximum(T1, T0))
            k = int(np.argmax(er))
            res[dV] = dict(ERmax=float(er[k]), IL=float(il[k]), minIL_4dB=float(il[er >= 4].min()))
        shift = float(LAM0 * (rm.dn(6.0) - rm.dn(0.0)) / rm.ng * 1e12)
        out[name] = {
            "model": dict(Q=LAM0 / rm.fwhm(), fwhm_pm=rm.fwhm() * 1e12, ER6=res[6.0]["ERmax"], IL6=res[6.0]["IL"],
                          IL4dB4V=res[4.0]["minIL_4dB"], dinv=float(rm.predicted_dinv_per_V())),
            "measured": m,
            "calibrated_input": dict(dneff_per_V_at_minus1V=KARIMELAHI[name]["dneff_per_V"], shift_pm_0_to_6V=shift),
        }
    return out


def alqadasi_bits(P, R=1.2, RL=50, Id=35e-9, DR=10e9, RIN_dB=-140):
    be = DR / math.sqrt(2)
    s1 = math.sqrt(2 * QE * (R * P + Id) + 4 * KB * TK / RL + (R * P) ** 2 * db2lin(RIN_dB))
    s0 = math.sqrt(2 * QE * Id + 4 * KB * TK / RL)
    return (20 * math.log10(R * P / ((s1 + s0) * math.sqrt(be))) - 1.76) / 6.02


def alqadasi_p_for_bits(n, **kw):
    lo, hi = 1e-9, 1.0
    for _ in range(200):
        m = math.sqrt(lo * hi)
        lo, hi = (lo, m) if alqadasi_bits(m, **kw) >= n else (m, hi)
    return hi


def alqadasi_mrr_laser(N, p_pd):
    """Eq. 13 with Table III values; P_laser taken as optical power (wall-plug factor applied separately)."""
    l = lambda x: db2lin(-x)  # noqa: E731
    num = 10 ** (0.3 * N * 0.020 / 10) * N
    den = l(0.0) * l(1.6) * l(4.0) * l(0.01) ** (N - 1) * l(0.01) ** math.log2(N)
    den2 = l(0.01) * l(0.01) ** (N - 1) * l(4.8)
    return num / den * p_pd / den2


def validate_alqadasi():
    p1 = alqadasi_p_for_bits(1, R=1.2)
    Ns = list(range(2, 201))
    plaser = [10 * math.log10(alqadasi_mrr_laser(N, p1) * 1e3) for N in Ns]
    nmax = max(N for N, p in zip(Ns, plaser) if p <= 10.0)
    p_pd = {n: 10 * math.log10(alqadasi_p_for_bits(n, R=1.2) * 1e3) for n in (1, 2, 3, 4, 5, 6)}
    return {"N": Ns, "laser_dBm_binary": plaser, "N_max_at_10dBm": nmax, "paper_N_max": 85,
            "pd_power_dBm_for_bits": p_pd, "responsivity_A_per_W": 1.2}


def pdk_summary():
    wl = np.array([LAM0])
    yb = ybranch(wl)
    T = bdc_T(wl)
    th = np.linspace(0, 2 * np.pi, 721)
    c = mzi_coupler(np.full(th.shape, LAM0), th)
    return {
        "y_branch_port1_to_port2_dB": float(10 * np.log10(np.abs(yb[("port_1", "port_2")][0]) ** 2)),
        "bdc_split_dB": [float(10 * np.log10(np.abs(T[0, 0][0]) ** 2)), float(10 * np.log10(np.abs(T[1, 0][0]) ** 2))],
        "grating_coupler_dB": float(10 * np.log10(np.abs(gc_t(wl)[0]) ** 2)),
        "halfring_gap150_R10_kappa2": float(halfring_kappa2(wl)[0]),
        "mzi_coupler_drop_range": [float(np.min(np.abs(c[1]) ** 2)), float(np.max(np.abs(c[1]) ** 2))],
        "loop_mirror_reflection_dB": float(10 * np.log10(np.abs(loop_mirror(wl)[0]) ** 2)),
        "ring_C_thermal_shift_pm_per_K": DT_PM_PER_K,
    }


# ================================================================== figures
def figures(res):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    os.makedirs(os.path.join(HERE, "figures"), exist_ok=True)
    ink, mut = "#0a1013", "#8b9ea5"
    plt.rcParams.update({"figure.facecolor": ink, "axes.facecolor": ink, "axes.edgecolor": "#33464f", "axes.labelcolor": mut,
                         "xtick.color": mut, "ytick.color": mut, "text.color": "#e3ebed", "font.size": 10,
                         "axes.grid": True, "grid.color": "#1c2a31"})
    sw = res["half_adder"]["xor_phase_sweep"]
    fig, ax = plt.subplots(figsize=(6, 3.4))
    ax.plot(sw["phase_deg"], sw["one_open_uA"], color="#e5503f", label="one arm open (sum = 1)")
    ax.plot(sw["phase_deg"], sw["both_open_uA"], color="#2c9fd6", label="both arms open (sum = 0)")
    ax.axhline(sw["threshold_uA"], color=mut, lw=1, label="threshold")
    tol = res["half_adder"]["xor_phase_tolerance_deg"]
    ax.axvspan(-tol, tol, color="#eba340", alpha=.12, label=f"±{tol:.0f}° keeps Q ≥ 7")
    ax.set_xlabel("XOR bias phase error (degrees)"); ax.set_ylabel("sum photocurrent (µA)")
    ax.legend(frameon=False, fontsize=8); fig.tight_layout()
    fig.savefig(os.path.join(HERE, "figures", "xor_phase_tolerance.png"), dpi=160); plt.close(fig)
    t = res["tile"]
    fig, ax = plt.subplots(figsize=(6, 3.4))
    for k, col in zip(("1", "2.5", "5"), ("#8fd3f0", "#3fa3d6", "#1f6fa3")):
        ax.semilogx(t["power_mW"], t["enob_by_rate"][k], color=col, label=f"{k} GS/s")
    ax.set_xlabel("laser power per comb line, in fibre (mW)"); ax.set_ylabel("effective bits")
    ax.legend(frameon=False, fontsize=8); fig.tight_layout()
    fig.savefig(os.path.join(HERE, "figures", "tile_effective_bits.png"), dpi=160); plt.close(fig)
    v = res["validation"]["alqadasi"]
    fig, ax = plt.subplots(figsize=(6, 3.4))
    ax.plot(v["N"], v["laser_dBm_binary"], color="#3fa3d6", label="our implementation of Eq. 8 + Eq. 13")
    ax.axhline(10, color=mut, lw=1, label="10 dBm laser")
    ax.axvline(85, color="#eba340", lw=1, label="paper: ~85 × 85")
    ax.set_xlabel("matrix size N"); ax.set_ylabel("required laser power (dBm)")
    ax.legend(frameon=False, fontsize=8); fig.tight_layout()
    fig.savefig(os.path.join(HERE, "figures", "alqadasi_reproduction.png"), dpi=160); plt.close(fig)


def main():
    tile = Tile()
    res = {
        "sources": SOURCES,
        "pdk": pdk_summary(),
        "validation": {"rings": validate_rings(), "alqadasi": validate_alqadasi()},
        "half_adder": half_adder_report(),
        "tile": tile_report(tile),
        "lighting": lighting_report(tile),
    }
    with open(os.path.join(HERE, "results.json"), "w") as f:
        json.dump(res, f, indent=1, default=float)
    figures(res)
    return res


if __name__ == "__main__":
    r = main()
    print("PDK:", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in r["pdk"].items()})
    for k, v in r["validation"]["rings"].items():
        print("ring", k, {a: round(b, 2) for a, b in v["model"].items()}, "| measured", v["measured"])
    a = r["validation"]["alqadasi"]
    print("Al-Qadasi: N_max", a["N_max_at_10dBm"], "(paper 85); PD dBm per bits", {k: round(v, 1) for k, v in a["pd_power_dBm_for_bits"].items()})
    ha = r["half_adder"]
    print("half adder budget", {k: round(v, 2) for k, v in ha["budget"].items()}, "detuning pm", round(ha["laser_detuning_pm"], 1))
    for row in ha["levels"]:
        print(f"  A={row['a']} B={row['b']} carry {row['carry_uA']:8.3f} ±{row['carry_sigma_uA']:.3f}  sum {row['sum_uA']:8.4f} ±{row['sum_sigma_uA']:.3f}")
    d = ha["decisions"]
    print(f"  carry Q {d['carry']['Q']:.1f} thr {d['carry']['threshold_uA']:.2f} | sum Q {d['sum']['Q']:.1f} thr {d['sum']['threshold_uA']:.3f} | tol ±{ha['xor_phase_tolerance_deg']:.0f} deg | min fibre {ha['min_fibre_power_uW_for_Q7']:.0f} uW")
    cq = [c["min_Q"] for c in ha["corners"]]
    print(f"  corners (heaters re-trimmed): min Q {min(cq):.1f}..{max(cq):.1f}; nominal heaters fail at",
          sum(c["min_Q_with_nominal_heaters"] < 7 for c in ha["corners"]), "of", len(cq), "corners")
    print("  full solution / single pass:", [(round(x["carry_ratio"], 3), round(x["sum_ratio"], 3)) for x in ha["single_pass_vs_full"]])
    t = r["tile"]
    print("tile T range", [round(v, 3) for v in t["modulator_T_range"]], "g", round(t["weight_scale_g"], 4), "drop range", [round(v, 3) for v in t["coupler_drop_range"]], "demux xtalk dB", round(t["demux_worst_crosstalk_dB"], 1))
    for k, v in t["enob_by_rate"].items():
        print(f"  {k} GS/s ENOB:", " ".join(f"{e:.2f}" for e in v[::3]))
    print("  design (2 mW, 5 GS/s) ENOB", round(t["design_enob"], 2), "budget", {k: round(v, 2) for k, v in t["enob_with_one_impairment"].items()})
    li = r["lighting"]
    print("lighting sigma_w", round(li["sigma_w"], 4), "validation", {k: {a: round(b, 4) for a, b in v.items()} for k, v in li["validation"].items()})
