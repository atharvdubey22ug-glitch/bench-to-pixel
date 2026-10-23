#!/usr/bin/env python3
"""Export the mesh chip's simulated behaviour for the web page (../mesh/data.json).

The page runs the digit network in the browser twice: in software (mesh_net.py's rows) and on the
simulated chip. For the chip, every page of the example die (mesh_results.json, nominal corner) is
rebuilt from its stored heater settings, and its scaled readings on 3,000 training digits (no noise)
are fitted with a quadratic form x^T Q x per detector (page_Q). That carries what the chip does to a
layer (crosstalk, coupler errors, reflections, the residual of calibration) except the small part of
the encoders' response that is not quadratic; check_quadratic() measures what is left out, on 2,000
test digits, against the full simulation.

Also exported: the feature map, the network, 24 test digits, the corner study, the speed study,
chip facts, and a check list (window.__check on the page reproduces these numbers).
"""
import base64
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import mesh_sim as M  # noqa: E402
import mesh_net as MN  # noqa: E402
import full_adder_sim as F  # noqa: E402

OUT = os.path.join(HERE, "..", "mesh", "data.json")


def rebuild_example(res, net):
    ex = res["example"]
    d = ex["detail"]
    circ = M.Circuit(tuple(ex["corner"]))
    X = M.crosstalk_matrix(circ)
    die = M.Die(circ, X, np.random.default_rng(ex["seed"]))
    assert np.allclose(die.eta, d["eta"]), "die does not match the stored example"
    enc = d["encoders"]
    pages = [dict(P=np.asarray(P) * 1e-3, scale=np.asarray(sc), n=n) for P, sc, n in
             zip(d["page_P_mW"], d["page_scale"], [len(p) for p in list(net["pages1"]) + list(net["pages2"])])]
    return die, enc, pages


def page_Q(die, enc, pages, net, A_fit):
    """Per page, the quadratic form that best matches the simulated chip's scaled readings on the inputs
    it actually sees: the features of 3,000 training digits (layer 1), and the electronics' outputs
    computed from the chip's own layer-1 readings of those digits (layer 2). Least squares over the 36
    products x_i x_j."""
    iu = np.triu_indices(M.N)
    full = M.chip_layer(die, enc, pages, truth=True)
    _, _, z, _ = MN.forward(net, A_fit, full)
    n1 = len(net["pages1"])
    out = []
    for k, pg in enumerate(pages):
        X = A_fit if k < n1 else z
        R = full(k, None, X)                                       # (B, n): scaled readings
        F = X[:, iu[0]] * X[:, iu[1]] * np.where(iu[0] == iu[1], 1.0, 2.0)[None]
        coef = np.linalg.lstsq(F, R, rcond=None)[0]                # (36, n)
        Q = np.zeros((pg["n"], M.N, M.N))
        for d in range(pg["n"]):
            Q[d][iu] = coef[:, d]
            Q[d] = Q[d] + np.triu(Q[d], 1).T
        out.append(Q)
    return out


def chip_quadratic(Qs, net, A):
    def layer(k, rows, inp):
        return np.einsum("bi,dij,bj->bd", inp, Qs[k], inp)
    return MN.forward(net, A, layer)


def check_quadratic(die, enc, pages, Qs, net, A):
    full = MN.forward(net, A, M.chip_layer(die, enc, pages, truth=True))
    quad = chip_quadratic(Qs, net, A)
    h_err = float(np.sqrt(np.mean((full[1] - quad[1]) ** 2)) / np.sqrt(np.mean(full[1] ** 2)))
    same = float(np.mean(np.argmax(full[0], 1) == np.argmax(quad[0], 1)))
    return dict(hidden_rms_difference=h_err, same_answer=same, digits=len(A))


def r(a, n):
    return np.round(np.asarray(a, float), n).tolist()


def main():
    res = json.load(open(os.path.join(HERE, "mesh_results.json")))
    net, fm, info = MN.load()
    _, _, Xte, yte = MN.mnist()
    die, enc, pages = rebuild_example(res, net)
    Xtr = MN.mnist()[0]
    A_fit = MN.features(fm, Xtr[np.random.default_rng(2).choice(len(Xtr), 3000, replace=False)])
    Qs = page_Q(die, enc, pages, net, A_fit)
    A = MN.features(fm, Xte)
    chk = check_quadratic(die, enc, pages, Qs, net, A[:2000])
    print("quadratic form vs full simulation:", chk)
    quad = chip_quadratic(Qs, net, A)
    soft = MN.forward(net, A)
    acc_q = float(np.mean(np.argmax(quad[0], 1) == yte))
    # 24 test digits: two of each class, then four the chip and the software disagree on (if any)
    rng = np.random.default_rng(1)
    pick = [int(i) for c in range(10) for i in rng.choice(np.where(yte == c)[0], 2, replace=False)]
    dis = np.where(np.argmax(quad[0], 1) != np.argmax(soft[0], 1))[0]
    pick += [int(i) for i in dis[:4]]
    imgs = [base64.b64encode((Xte[i] * 255).round().astype(np.uint8).tobytes()).decode() for i in pick]
    spd = json.load(open(os.path.join(HERE, "speed", "mesh_speed_results.json")))
    rep = json.load(open(os.path.join(HERE, "klayout", "mesh_build_report.json")))
    cx = lambda a: [r(np.real(a), 5), r(np.imag(a), 5)]  # noqa: E731
    ex = res["example"]
    data = dict(
        features=dict(mean=r(fm["mean"], 4), comp=r(fm["comp"].T, 5), sd=r(fm["sd"], 4), scale=MN.SC),
        net=dict(pages1=[cx(p) for p in net["pages1"]], pages2=[cx(p) for p in net["pages2"]],
                 a=r(net["a"], 5), b=r(net["b"], 5), k=float(net["k"]), c=r(net["c"], 5)),
        chip=dict(Q=[r(q, 6) for q in Qs], corner=ex["corner"], seed=ex["seed"], accuracy_full=ex["accuracy"],
                  agreement_full=ex["agreement"], accuracy_quadratic=acc_q, quadratic_check=chk),
        digits=dict(idx=pick, label=[int(yte[i]) for i in pick], img=imgs),
        corners=[dict(corner=c["corner"], acc=[c["accuracy_min"], c["accuracy_mean"], c["accuracy_max"]],
                      agree=[c["agreement_min"], c["agreement_mean"]], hidden=c["hidden_rms_error_mean"],
                      row=[c["row_error_median"], c["row_error_worst"]], fits=c["fits_per_page_mean"]) for c in res["corners"]],
        dies=[dict(c=d["corner"], a=d["accuracy"], g=d["agreement"]) for d in res["dies"]],
        software=dict(acc=info["accuracy"]["software_test"], linear=info["accuracy"]["linear_test"], mlp=info["accuracy"]["mlp_test"]),
        summary=res["summary"], die_model=res["die_model"], model_check=res["model_check"],
        speed=dict(dc=spd["dc"], latency=spd["latency"], summary=spd["summary"],
                   runs=[dict(io=x["io"], al=x["aligned"], rate=x["rate_GBd"], err=x["layer1_error"], acc=x["accuracy_noiseless"],
                              acc10=x["accuracy_10mW"], acc100=x["accuracy_100mW"], agree=x["agreement_noiseless"],
                              agree10=x["agreement_10mW"], agree100=x["agreement_100mW"]) for x in spd["runs"]]),
        facts=dict(die_um=rep["die_um"], components=rep["components"], heaters=rep["heaters"], pads=rep["pads"],
                   mzis=len(rep["mzis"]), channel_um=rep["channel_lengths_um"]["ch0"], encoder_pitch_um=rep["encoder_pitch_um"],
                   measurements_per_die=float(np.mean([d["measurements_calibration"] + d["measurements_programming"] for d in res["dies"]]))),
    )
    # check list for the page: features, software and chip outputs of three digits
    chk_idx = pick[:3]
    data["check"] = [dict(i=int(i), x=r(A[i], 6), h_soft=r(soft[1][i], 6), h_chip=r(quad[1][i], 6),
                          logit_soft=r(soft[0][i], 5), logit_chip=r(quad[0][i], 5)) for i in chk_idx]
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(data, open(OUT, "w"), separators=(",", ":"))
    print("wrote", OUT, os.path.getsize(OUT), "bytes; accuracy through the quadratic forms", acc_q)


if __name__ == "__main__":
    main()
