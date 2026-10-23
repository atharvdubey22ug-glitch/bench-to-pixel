#!/usr/bin/env python3
"""
How fast can the mesh chip run the digit network? Streams of test digits drive the eight input encoders
of klayout/EBeam_heaters_atharvdubey_Mesh8, one digit per symbol slot with all eight inputs in parallel,
while the mesh holds one page of the network; the three detector outputs are read straight off the
chip (time domain, td.py: the SiEPIC extraction as a delay-line scattering network, back-reflections
included, as in the XOR, parity and neuron studies).

The chip: the nominal corner with no fabrication phase errors and no crosstalk, so it can be
programmed exactly (the frequency-domain model of mesh_sim.py, noiseless measurements). Every page of
the network is run: layer 1's three pages on the digits' features, then layer 2's four pages on the
values the electronics compute from layer 1's sampled outputs (z = clip(a h + b, 0, 1)), as a real
two-pass system would.

Encoders are treated as push-pull phase modulators of the drawn placement (the TiW heaters on this
process are far too slow; a modulator process would put fast phase sections there), with ideal,
100 GHz or 50 GHz bandwidth and the same bandwidth on the photodiodes. Each heater's drive is delayed
by its position, so every input's change reaches the mesh at the same time ("aligned"); "unaligned"
drives them all at once.

Measures, per symbol rate:
  error      RMS difference between the sampled outputs and the chip's own slow (DC) outputs, over the
             wanted values' RMS, at the best sampling instant (layer 1, pages 1 to 3, no noise)
  accuracy   the network's answers at speed on 256 test digits, against the labels and against the
             slow chip's answers: without noise (what the bandwidth costs), and with shot noise and a
             20 pA/rtHz amplifier on each photodiode for 10 mW and 100 mW of laser into the input fibre
             (the noise bandwidth is the photodiodes', or 0.75 x the symbol rate with ideal ones)
  latency    from the first encoder's change to the mesh output and to the output fibre
Summary (summarize()): the highest rate at which the network still gives the slow chip's answer on at
least 95% of the digits, without noise and at each laser power.

Outputs: mesh_speed_results.json, mesh_eyes.npz.
"""
import json
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, ".."))
from td import TDCircuit, F, _wg  # noqa: E402
from eyes import gfilter  # noqa: E402
import mesh_sim as M  # noqa: E402
import mesh_net as MN  # noqa: E402

RATES_IDEAL = [25e9, 50e9, 100e9, 200e9, 400e9]
RATES_IO = [25e9, 50e9, 75e9, 100e9, 125e9, 150e9, 200e9]
IO = {"ideal": None, "silicon_50GHz": 50e9, "lithium_niobate_100GHz": 100e9}
TIA, NIMG = 20e-12, 256
P_LIST = [10e-3, 100e-3]                  # laser power into the input fibre
DT = 0.02e-12


def ideal_die():
    """Nominal corner, no phase errors, no crosstalk, nominal heater efficiency: what td.py models."""
    c = M.Circuit()
    rng = np.random.default_rng(0)
    die = M.Die(c, np.eye(len(c.wg))[:, [c.wg_role[h] for h in M.HEATERS]], rng)
    die.eta[:] = math.pi / M.P_PI
    die.phi0[:] = 0.0
    die.fs = die._full_scale()
    die.measure = die.power
    die.page_measure = die.page_power
    return die


def program(die, net, rng):
    enc = M.calibrate_encoders(die, rng)
    pages = M.program_network(die, enc, net, rng)
    return enc, pages


def run_fast(tc, n, static, dyn, src_port, record):
    """td.TDCircuit.run with constant heaters folded into the links and the time-varying ones
    precomputed per link (same physics, much faster)."""
    R = tc.R
    hist = np.zeros((R, tc.P), complex)
    fac0 = tc.lF.copy()
    dyn_links = {}
    for i, r, d in tc.heat_terms:
        if r in dyn:
            dyn_links.setdefault(i, []).append((r, d))
        else:
            fac0[i] *= np.exp(1j * static.get(r, 0.0))
    li = np.array(sorted(dyn_links))
    E = np.ones((li.size, n), complex)
    for k, i in enumerate(li):
        ph = np.zeros(n)
        for r, d in dyn_links[i]:
            w = np.asarray(dyn[r], float)
            ph[d:] += w[:n - d]
            ph[:d] += w[0]
        E[k] = np.exp(1j * ph)
    out = {k: np.zeros(n, complex) for k in record}
    rec = {k: tc.fibre[k] for k in record}
    lq, lp, Ds = tc.lq, tc.lp, tc.Dsteps
    S = tc.S
    for t in range(n):
        fac = fac0.copy()
        fac[li] *= E[:, t]
        vin = np.zeros(tc.P, complex)
        np.add.at(vin, lq, fac * hist[(t - Ds) % R, lp])
        vin[src_port] += 1.0
        vout = S @ vin
        hist[t % R] = vout
        for k, pi in rec.items():
            out[k][t] = vout[pi]
    return out


def heater_delays(tc):
    """Steps from each encoder heater to the mesh: from the heater to the end of its arm (forward link)
    plus the link from the encoder's output Y-branch to the first mesh component."""
    comps = {c["idx"]: c for c in json.load(open(M.EXTRACT))["components"]}
    port_comp = lambda p: comps[tc.ports[p][0]]  # noqa: E731
    out = {}
    for i, r, d in tc.heat_terms:
        if not r.startswith("E"):
            continue
        L = tc.links[i]
        q = port_comp(L["q"])
        p = port_comp(L["p"])
        if q["center"][0] <= p["center"][0]:
            continue                                    # backward link
        # the next link: from q's opt1 port (the Y-branch output) onward
        nxt = [l for l in tc.links if l["p"] != L["q"] and tc.ports[l["p"]][0] == q["idx"] and tc.ports[l["p"]][1] == "port_1"]
        out[r] = d + int(round(nxt[0]["D"] / tc.dt))
    assert len(out) == 16, out
    return out


def drive_wave(values, rate, dt, n, start, bw):
    """Piecewise-constant per-symbol values (held for a symbol), filtered by the modulator bandwidth."""
    t = (np.arange(n) - start) * dt
    k = np.floor(t * rate).astype(int)
    v = np.where(k < 0, values[0], np.where(k >= values.size, values[-1], values[np.clip(k, 0, values.size - 1)]))
    return gfilter(v.astype(float), dt, bw)


def run_page(tc, enc, page, x, rate, bw, dly, aligned=True, tail_ps=500.0):
    """x: (n_sym x 8) inputs. Returns detector powers (fraction of laser power) sampled per symbol at the
    best instant (n_sym x 3), the folded traces for eyes, and that instant."""
    dt = tc.dt
    spb = 1.0 / rate / dt
    lead = int(round(200e-12 / dt))
    dmax = max(dly.values())
    n = lead + int(math.ceil((len(x) + 4) * spb)) + dmax + int(round(tail_ps * 1e-12 / dt))
    Pe = M.encoder_powers(enc, x)                                      # (n_sym, 16)
    eta = math.pi / M.P_PI
    dyn = {}
    for k, r in enumerate(M.ENC_H):
        skew = dmax - dly[r] if aligned else 0
        dyn[r] = drive_wave(Pe[:, k] * eta, rate, dt, n, lead + skew, bw)
    static = {r: float(page["P"][k] * eta) for k, r in enumerate(M.MESH_H)}
    e = run_fast(tc, n, static, dyn, tc.fibre["laser"], M.DETS)
    p = np.stack([gfilter(np.abs(e[d]) ** 2, dt, bw) for d in M.DETS], 1)
    return p, lead, spb, n


def sample(p, want, lead, spb, n_sym, skip=3):
    """Best delay and sampling phase: the instant where the sampled outputs best match `want`."""
    best = None
    for dly in range(0, int(2 * spb) + int(round(500e-12 / DT)), max(1, int(spb / 16))):
        idx = (lead + dly + (np.arange(n_sym) + 0.5) * spb).astype(int)
        if idx[-1] >= len(p):
            break
        v = p[idx]
        err = np.sqrt(np.mean((v[skip:] - want[skip:]) ** 2)) / np.sqrt(np.mean(want[skip:] ** 2))
        if best is None or err < best[0]:
            best = (err, dly, idx)
    return best


def noisy(v, rate, bw, rng, p_laser):
    """Photodiode noise on powers v (fraction of laser power): shot noise and amplifier noise."""
    bw_n = bw if bw else 0.75 * rate
    i = F.DET["resp"] * p_laser * v
    sig = np.sqrt((2 * F.P.QE * np.maximum(i, 0) + 2 * TIA ** 2) * bw_n)
    return (i + sig * rng.standard_normal(i.shape)) / (F.DET["resp"] * p_laser)


def main():
    t0 = time.time()
    net, fm, info = MN.load()
    _, _, Xte, yte = MN.mnist()
    rng = np.random.default_rng(3)
    sel = rng.choice(len(Xte), NIMG, replace=False)
    A, y = MN.features(fm, Xte[sel]), yte[sel]
    die = ideal_die()
    DIE.append(die)
    enc, pages = program(die, net, rng)
    print("programmed", [round(p["info"]["true_row_error"], 4) for p in pages], f"{time.time() - t0:.0f}s", flush=True)
    # DC (slow) reference: the same chip, noiseless
    dc_layer = M.chip_layer(die, enc, pages, truth=True)
    logit_dc, h_dc, z_dc, s_dc = MN.forward(net, A, dc_layer)
    pred_dc = np.argmax(logit_dc, 1)
    tc = TDCircuit(extract=M.EXTRACT, dt=DT)
    dly = heater_delays(tc)
    res = {"method": __doc__.strip().splitlines()[0], "n_digits": NIMG, "laser_mW": [p * 1e3 for p in P_LIST],
           "dc": dict(accuracy=float(np.mean(pred_dc == y)), agreement_with_software=float(np.mean(pred_dc == np.argmax(MN.forward(net, A)[0], 1)))),
           "skew_ps": {r: round((max(dly.values()) - d) * tc.dt * 1e12, 2) for r, d in sorted(dly.items())},
           "page_row_errors": [round(p["info"]["true_row_error"], 4) for p in pages]}
    # latency: step the first encoder (largest delay to the mesh) and watch output 7
    res["latency"] = latency(tc, enc, pages[0], dly)
    print("latency", res["latency"], flush=True)
    jobs = []
    for io, bw in IO.items():
        for rate in (RATES_IDEAL if bw is None else RATES_IO):
            jobs.append((io, bw, rate, True))
        for rate in (50e9, 100e9):
            jobs.append((io, bw, rate, False))
    import pickle
    import multiprocessing as mp
    state = os.path.join(HERE, ".mesh_speed_state.pkl")
    pickle.dump(dict(enc=enc, pages=pages, sel=sel, pred_dc=pred_dc), open(state, "wb"))
    runs, eyes = [], {}
    with mp.get_context("spawn").Pool(int(os.environ.get("MESH_PROCS", "2")), initializer=_init, initargs=(state,)) as pool:
        for (io, bw, rate, aligned), (r, folds) in zip(jobs, pool.imap(_job, jobs)):
            r.update(io=io, aligned=aligned, bw_GHz=bw / 1e9 if bw else None)
            runs.append(r)
            if aligned:
                eyes[f"{io}|{int(rate / 1e9)}"] = folds
            print(io, int(rate / 1e9), "aligned" if aligned else "unaligned", round(r["layer1_error"], 4),
                  {k: v for k, v in r.items() if k.startswith("acc")}, f"{time.time() - t0:.0f}s", flush=True)
    os.remove(state)
    res["runs"] = runs
    summ = summarize(res)
    res["summary"] = summ
    print(json.dumps(summ))
    with open(os.path.join(HERE, "mesh_speed_results.json"), "w") as f:
        json.dump(res, f, indent=1, default=float)
    np.savez_compressed(os.path.join(HERE, "mesh_eyes.npz"), **{k.replace("|", "__"): v[:64].astype(np.float16) for k, v in eyes.items()})
    print(f"done {time.time() - t0:.0f}s")


def summarize(res):
    """Highest rate (aligned drives) at which the network still gives the slow chip's answer on at least
    95% of the digits, without noise and at each laser power; and where layer 1's error is below 5%."""
    summ = {}
    for io in IO:
        rs = [r for r in res["runs"] if r["io"] == io and r["aligned"]]
        summ[io] = dict(error_below_5pct=max([r["rate_GBd"] for r in rs if r["layer1_error"] < 0.05], default=0),
                        **{f"agree95_{k}": max([r["rate_GBd"] for r in rs if r[f"agreement_{k}"] >= 0.95], default=0)
                           for k in ["noiseless"] + [f"{p * 1e3:.0f}mW" for p in P_LIST]})
    return summ


def network_at(tc, enc, pages, net, A, y, pred_dc, rate, bw, dly, aligned, seed=0):
    """The whole network with every page run in the time domain at this symbol rate: layer 1 once,
    then layer 2 for each laser power (its inputs are computed from layer 1's noisy readings)."""
    n1 = len(net["pages1"])
    dc = M.chip_layer(DIE[0], enc, pages, truth=True)

    def page_at(k, inp):
        pg = pages[k]
        p, lead, spb, n = run_page(tc, enc, pg, inp, rate, bw, dly, aligned)
        want = dc(k, None, inp) * pg["scale"][None, :pg["n"]]
        err, dly_s, idx = sample(p[:, :pg["n"]], want, lead, spb, len(inp))
        return p[idx][:, :pg["n"]], err, (p, idx, spb, want)

    l1 = [page_at(k, A) for k in range(n1)]
    errs = [e for _, e, _ in l1]
    p, idx, spb, want = l1[0][2]
    seg = int(round(2 * spb))
    t0_ = idx[0] - int(spb / 2)
    folds = np.array([p[t0_ + int(j * spb) - seg // 4: t0_ + int(j * spb) - seg // 4 + seg, 0] for j in range(4, len(A) - 4)
                      if t0_ + int(j * spb) - seg // 4 + seg <= len(p)]) / max(np.abs(want[:, 0]).max(), 1e-30)
    out = dict(rate_GBd=rate / 1e9, layer1_error=float(np.max(errs)))
    for pw in [None] + P_LIST:
        rng = np.random.default_rng(seed)
        rd = (lambda v: v) if pw is None else (lambda v: noisy(v, rate, bw, rng, pw))
        h = np.concatenate([rd(v) / pages[k]["scale"][None, :pages[k]["n"]] for k, (v, _, _) in enumerate(l1)], 1)
        z = np.concatenate([np.clip(net["a"] * h + net["b"], 0, 1), np.ones((len(A), 1))], 1)
        s2 = np.concatenate([rd(page_at(n1 + k, z)[0]) / pages[n1 + k]["scale"][None, :pages[n1 + k]["n"]] for k in range(len(net["pages2"]))], 1)
        pred = np.argmax(net["k"] * s2 + net["c"], 1)
        key = "noiseless" if pw is None else f"{pw * 1e3:.0f}mW"
        out[f"accuracy_{key}"] = float(np.mean(pred == y))
        out[f"agreement_{key}"] = float(np.mean(pred == pred_dc))
    return out, folds[:, np.linspace(0, folds.shape[1] - 1, 400).round().astype(int)].astype(np.float32)


def latency(tc, enc, page, dly):
    """Step encoder inputs from all -1 to all +1 at t = 0 (aligned); time for output 7 to reach half
    its change at the fibre, and minus the output route's delay, at the mesh output."""
    rate = 1e9
    x = np.array([[-1.0] * 7 + [1.0], [1.0] * 8])
    p, lead, spb, n = run_page(tc, enc, page, np.repeat(x, [1, 3], axis=0), rate, None, dly, True, tail_ps=0)
    v = p[:, 0]
    a, b = v[lead + int(spb) - 5], v[-1]
    t_change = lead + int(spb)              # the encoder farthest from the mesh changes first (skew 0)
    k = np.argmax(np.abs(v[t_change:] - a) > 0.5 * abs(b - a)) + t_change
    fib = (k - t_change) * tc.dt
    rep = json.load(open(os.path.join(HERE, "..", "klayout", "mesh_build_report.json")))
    _, tg = _wg(rep["lengths_um"]["route_o7"])
    return dict(to_fibre_ps=round(fib * 1e12, 1), output_route_ps=round(tg * 1e12, 1), to_mesh_output_ps=round((fib - tg) * 1e12, 1))


DIE = []
_W = {}


def _init(state):
    import pickle
    st = pickle.load(open(state, "rb"))
    net, fm, info = MN.load()
    _, _, Xte, yte = MN.mnist()
    DIE.append(ideal_die())
    tc = TDCircuit(extract=M.EXTRACT, dt=DT)
    _W.update(st, net=net, A=MN.features(fm, Xte[st["sel"]]), y=yte[st["sel"]], tc=tc, dly=heater_delays(tc))


def _job(j):
    io, bw, rate, aligned = j
    w = _W
    return network_at(w["tc"], w["enc"], w["pages"], w["net"], w["A"], w["y"], w["pred_dc"], rate, bw, w["dly"], aligned)

if __name__ == "__main__":
    if sys.argv[1:] == ["summary"]:              # recompute the summary of an existing results file
        f = os.path.join(HERE, "mesh_speed_results.json")
        res = json.load(open(f))
        res["summary"] = summarize(res)
        json.dump(res, open(f, "w"), indent=1, default=float)
        print(json.dumps(res["summary"]))
    else:
        main()
