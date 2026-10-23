#!/usr/bin/env python3
"""
The digit classifier the mesh chip runs (mesh_sim.py), trained in software with the chip's structure.

Data: MNIST (60,000 training and 10,000 test digits, 28 x 28 pixels), downloaded once to data/mnist.npz.
Features (electronics): the first 7 principal components of the training images, scaled so 2.5
standard deviations is 1 and clipped to [-1, 1]. With a constant 1 as the eighth input (the bias,
which lets the square-law layers form terms linear in the features) they are the chip's input x.

Layer 1 (chip): seven outputs h_k = |t_k . x|^2, read three at a time: pages of 3, 3 and 1 rows, each
page the first rows of an 8x8 unitary (what one setting of the mesh can give).
Between layers (electronics): z_k = clip(a_k h_k + b_k, 0, 1), seven scales and offsets; z = [z, 1].
Layer 2 (chip): ten outputs s_c = |u_c . z|^2, one per digit, pages of 3, 3, 3 and 1 rows.
Answer: the largest of k s_c + b_c (k > 0, ten offsets).

Baselines trained the same way on the same 7 features (+ bias) for context: a linear classifier and
an electronic network with one hidden layer of 7 ReLU units.

Writes mesh_net.json: the feature map, every page's rows, the electronic parameters, accuracies.
"""
import hashlib
import json
import os
import sys
import time
import urllib.request

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data", "mnist.npz")
URL = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz"
SHA256 = "731c5ac602752760c8e48fbffcf8c3b850d9dc2a2aedcf2cc48468fc17b673d1"
OUT = os.path.join(HERE, "mesh_net.json")
NF, SC = 7, 2.5
PAGES1, PAGES2 = (3, 3, 1), (3, 3, 3, 1)
EPOCHS, BATCH, LR = 60, 256, 1e-2


def mnist():
    if not os.path.exists(DATA):
        os.makedirs(os.path.dirname(DATA), exist_ok=True)
        urllib.request.urlretrieve(URL, DATA)
    assert hashlib.sha256(open(DATA, "rb").read()).hexdigest() == SHA256, "unexpected mnist.npz"
    d = np.load(DATA)
    return d["x_train"].reshape(-1, 784) / 255.0, d["y_train"], d["x_test"].reshape(-1, 784) / 255.0, d["y_test"]


def features(fm, X):
    """Images (B x 784) -> chip inputs (B x 8): 7 clipped principal components and the bias 1."""
    F = (X - fm["mean"]) @ fm["comp"] / fm["sd"]
    return np.concatenate([np.clip(F / SC, -1, 1), np.ones((len(X), 1))], 1)


def fit_features(Xtr):
    mean = Xtr.mean(0)
    w, V = np.linalg.eigh(np.cov((Xtr - mean).T))
    comp = V[:, ::-1][:, :NF]
    comp = comp * np.sign(comp.sum(0, keepdims=True) + 1e-12)          # fixed sign convention
    sd = ((Xtr - mean) @ comp).std(0)
    return dict(mean=mean, comp=comp, sd=sd)


def rows_of(pages):
    """List of unitaries' first rows -> (n_rows, 8) complex, in page order."""
    return np.concatenate([U[:n] for U, n in pages])


def forward(net, x, layer=None):
    """The network on chip inputs x (B x 8). layer(k, rows, inputs) -> intensities (B x len(rows)) for
    page k; default: exact |rows . inputs|^2 (the software network)."""
    if layer is None:
        def layer(k, rows, inp):
            return np.abs(inp @ rows.T) ** 2
    h = np.concatenate([layer(k, r, x) for k, r in enumerate(net["pages1"])], 1)
    z = np.concatenate([np.clip(net["a"] * h + net["b"], 0, 1), np.ones((len(x), 1))], 1)
    s = np.concatenate([layer(len(net["pages1"]) + k, r, z) for k, r in enumerate(net["pages2"])], 1)
    return net["k"] * s + net["c"], h, z, s


def load(path=OUT):
    d = json.load(open(path))
    cx = lambda v: np.asarray(v[0]) + 1j * np.asarray(v[1])  # noqa: E731
    net = dict(pages1=[cx(p) for p in d["pages1"]], pages2=[cx(p) for p in d["pages2"]],
               a=np.asarray(d["a"]), b=np.asarray(d["b"]), k=d["k"], c=np.asarray(d["c"]))
    fm = dict(mean=np.asarray(d["features"]["mean"]), comp=np.asarray(d["features"]["comp"]), sd=np.asarray(d["features"]["sd"]))
    return net, fm, d


def train():
    import jax
    import jax.numpy as jnp
    import optax
    jax.config.update("jax_enable_x64", True)
    Xtr, ytr, Xte, yte = mnist()
    fm = fit_features(Xtr)
    Atr, Ate = features(fm, Xtr), features(fm, Xte)

    def unitary(q):
        A = q[0] + 1j * q[1]
        return jax.scipy.linalg.expm((A - A.conj().T) / 2)

    def model(kind):
        def init(key):
            k = jax.random.split(key, 8)
            if kind == "linear":
                return dict(W=jax.random.normal(k[0], (8, 10)) * 0.3)
            if kind == "mlp":
                return dict(W1=jax.random.normal(k[0], (8, NF)) * 0.5, W2=jax.random.normal(k[1], (NF + 1, 10)) * 0.5)
            return dict(L1=[jax.random.normal(k[j], (2, 8, 8)) * 0.5 for j in range(3)],
                        L2=[jax.random.normal(k[3 + j], (2, 8, 8)) * 0.5 for j in range(4)],
                        a=jnp.full(NF, 4.0), b=jnp.full(NF, -0.5), k=jnp.array(20.0), c=jnp.zeros(10))

        def f(p, x):
            if kind == "linear":
                return x @ p["W"]
            if kind == "mlp":
                hid = jax.nn.relu(x @ p["W1"])
                return jnp.concatenate([hid, jnp.ones((x.shape[0], 1))], 1) @ p["W2"]
            h = jnp.concatenate([jnp.abs(x @ unitary(q)[:n].T) ** 2 for q, n in zip(p["L1"], PAGES1)], 1)
            z = jnp.concatenate([jnp.clip(p["a"] * h + p["b"], 0, 1), jnp.ones((x.shape[0], 1))], 1)
            s = jnp.concatenate([jnp.abs(z @ unitary(q)[:n].T) ** 2 for q, n in zip(p["L2"], PAGES2)], 1)
            return p["k"] * s + p["c"]
        return init, f

    out = {}
    for kind in ("linear", "mlp", "chip"):
        init, f = model(kind)
        p = init(jax.random.PRNGKey(0))
        steps = EPOCHS * (len(Atr) // BATCH)
        opt = optax.adam(optax.cosine_decay_schedule(LR, steps))
        st = opt.init(p)

        def loss(p, x, y):
            return optax.softmax_cross_entropy_with_integer_labels(f(p, x), y).mean()

        @jax.jit
        def step(p, st, x, y):
            l, g = jax.value_and_grad(loss)(p, x, y)
            u, st = opt.update(g, st, p)
            return optax.apply_updates(p, u), st, l
        rng = np.random.default_rng(0)
        t = time.time()
        for ep in range(EPOCHS):
            idx = rng.permutation(len(Atr))
            for k in range(0, len(idx) - BATCH + 1, BATCH):
                p, st, _ = step(p, st, Atr[idx[k:k + BATCH]], ytr[idx[k:k + BATCH]])
        acc = float(np.mean(np.argmax(np.asarray(f(p, Ate)), 1) == yte))
        acc_tr = float(np.mean(np.argmax(np.asarray(f(p, Atr)), 1) == ytr))
        print(kind, "test %.4f train %.4f (%.0f s)" % (acc, acc_tr, time.time() - t), flush=True)
        out[kind] = (p, acc, acc_tr)

    p, acc, acc_tr = out["chip"]
    U1 = [np.asarray(unitary(q)) for q in p["L1"]]
    U2 = [np.asarray(unitary(q)) for q in p["L2"]]
    net = dict(pages1=[U[:n] for U, n in zip(U1, PAGES1)], pages2=[U[:n] for U, n in zip(U2, PAGES2)],
               a=np.asarray(p["a"]), b=np.asarray(p["b"]), k=float(p["k"]), c=np.asarray(p["c"]))
    logits = forward(net, Ate)[0]
    acc_np = float(np.mean(np.argmax(logits, 1) == yte))
    assert abs(acc_np - acc) < 1e-3, (acc_np, acc)
    cl = lambda a: [np.real(a).tolist(), np.imag(a).tolist()]  # noqa: E731
    res = dict(method=__doc__.strip().splitlines()[0], data=dict(source=URL, sha256=SHA256, train=len(Atr), test=len(Ate)),
               features=dict(n=NF, scale_sd=SC, mean=fm["mean"].tolist(), comp=fm["comp"].tolist(), sd=fm["sd"].tolist(),
                             clipped_fraction=float(np.mean(np.abs(((Xtr - fm["mean"]) @ fm["comp"] / fm["sd"]) / SC) > 1))),
               training=dict(epochs=EPOCHS, batch=BATCH, lr=LR, optimizer="Adam, cosine decay", seed=0),
               pages1=[cl(r) for r in net["pages1"]], pages2=[cl(r) for r in net["pages2"]],
               a=net["a"].tolist(), b=net["b"].tolist(), k=net["k"], c=net["c"].tolist(),
               accuracy=dict(software_test=acc_np, software_train=acc_tr,
                             linear_test=out["linear"][1], mlp_test=out["mlp"][1]),
               multiply_adds=dict(features_electronic=784 * NF, chip=8 * 7 + 8 * 10, between_layers_electronic=2 * NF,
                                  electronic_mlp_same_size=8 * NF + (NF + 1) * 10))
    json.dump(res, open(OUT, "w"))
    return res


if __name__ == "__main__":
    r = train()
    print(json.dumps(r["accuracy"], indent=1))
