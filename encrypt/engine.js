/* Time-domain model of the single-pass optical XOR, ported from chip/speed/td.py.
   The circuit (circuit.json) is the SiEPIC extraction of klayout/EBeam_heaters_atharvdubey_FastXOR:
   every drawn waveguide is a delay line with its PDK group delay and complex transmission at 1550 nm,
   couplers / Y-branches / grating couplers keep their PDK S-matrices, and the two input switches are
   phase sections that change in time. Checked against the Python model (chip/speed/reference_run.json). */
(function (root) {
  "use strict";

  function build(c, dt) {
    const P = c.ports;
    const L = c.links.length;
    const lp = new Int32Array(L), lq = new Int32Array(L), D = new Int32Array(L);
    const Fre = new Float64Array(L), Fim = new Float64Array(L);
    const heat = [];
    let R = 0;
    c.links.forEach((l, i) => {
      lp[i] = l.p; lq[i] = l.q; Fre[i] = l.re; Fim[i] = l.im;
      D[i] = Math.max(1, Math.round(l.D / dt));
      R = Math.max(R, D[i]);
      l.heat.forEach(([role, d]) => heat.push({ i, role, d: Math.max(0, Math.round(d / dt)) }));
    });
    const nnz = c.S.length;
    const Sr = new Int32Array(nnz), Sc = new Int32Array(nnz), Sre = new Float64Array(nnz), Sim = new Float64Array(nnz);
    c.S.forEach(([r, col, re, im], k) => { Sr[k] = r; Sc[k] = col; Sre[k] = re; Sim[k] = im; });
    return { c, dt, P, L, lp, lq, D, Fre, Fim, heat, R: R + 2, Sr, Sc, Sre, Sim };
  }

  /* phases: role -> Float64Array(n) or number (rad, per pass). Laser is CW, amplitude 1, into the
     laser fibre port. Returns the complex field leaving the XOR fibre port. */
  function run(net, n, phases, outName) {
    const { P, L, lp, lq, D, Fre, Fim, heat, R, Sr, Sc, Sre, Sim } = net;
    const src = net.c.fibre.laser1, out = net.c.fibre[outName || "xor"];
    const hre = new Float64Array(R * P), him = new Float64Array(R * P);
    const vre = new Float64Array(P), vim = new Float64Array(P);
    const ore = new Float64Array(n), oim = new Float64Array(n);
    const ph = (role, k) => {
      const v = phases[role];
      if (v === undefined) return 0;
      return typeof v === "number" ? v : v[k < 0 ? 0 : k];
    };
    const hasHeat = new Uint8Array(L);
    heat.forEach(h => { hasHeat[h.i] = 1; });
    const fre = new Float64Array(L), fim = new Float64Array(L);
    for (let s = 0; s < n; s++) {
      fre.set(Fre); fim.set(Fim);
      if (heat.length) {
        const sum = new Float64Array(L);
        for (const h of heat) sum[h.i] += ph(h.role, s - h.d);
        for (let i = 0; i < L; i++) {
          if (!hasHeat[i]) continue;
          const c = Math.cos(sum[i]), si = Math.sin(sum[i]);
          const a = fre[i], b = fim[i];
          fre[i] = a * c - b * si; fim[i] = a * si + b * c;
        }
      }
      vre.fill(0); vim.fill(0);
      for (let i = 0; i < L; i++) {
        let row = (s - D[i]) % R; if (row < 0) row += R;
        const k = row * P + lp[i];
        const a = hre[k], b = him[k];
        vre[lq[i]] += fre[i] * a - fim[i] * b;
        vim[lq[i]] += fre[i] * b + fim[i] * a;
      }
      vre[src] += 1;
      const base = (s % R) * P;
      for (let p = 0; p < P; p++) { hre[base + p] = 0; him[base + p] = 0; }
      for (let k = 0; k < Sr.length; k++) {
        const a = vre[Sc[k]], b = vim[Sc[k]];
        hre[base + Sr[k]] += Sre[k] * a - Sim[k] * b;
        him[base + Sr[k]] += Sre[k] * b + Sim[k] * a;
      }
      ore[s] = hre[base + out]; oim[s] = him[base + out];
    }
    return { re: ore, im: oim };
  }

  function erf(x) {                      // Abramowitz-Stegun 7.1.26, |error| < 1.5e-7
    const s = x < 0 ? -1 : 1; x = Math.abs(x);
    const t = 1 / (1 + 0.3275911 * x);
    const y = 1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-x * x);
    return s * y;
  }

  /* NRZ drive level 0..1 at every time step; bit k spans [lead + k T, lead + (k+1) T).
     bw: modulator 3 dB bandwidth (Hz), Gaussian response; null = ideal (instant edges). */
  function drive(bits, rate, dt, n, lead, bw) {
    const out = new Float64Array(n);
    const T = 1 / rate;
    if (!bw) {
      for (let s = 0; s < n; s++) {
        const k = Math.floor((s - lead) * dt * rate);
        out[s] = k >= 0 && k < bits.length ? bits[k] : 0;
      }
      return out;
    }
    const sigma = Math.sqrt(Math.LN2) / (2 * Math.PI * bw);
    const edges = [];                     // (time, step) of every level change, starting from 0 before the data
    let prev = 0;
    for (let k = 0; k <= bits.length; k++) {
      const v = k < bits.length ? bits[k] : 0;
      if (v !== prev) edges.push([k * T, v - prev]);
      prev = v;
    }
    const reach = 6 * sigma;
    let first = 0, level = 0;
    for (let s = 0; s < n; s++) {
      const t = (s - lead) * dt;
      while (first < edges.length && edges[first][0] < t - reach) { level += edges[first][1]; first++; }
      let v = level;
      for (let j = first; j < edges.length && edges[j][0] < t + reach; j++)
        v += edges[j][1] * 0.5 * (1 + erf((t - edges[j][0]) / (Math.SQRT2 * sigma)));
      out[s] = Math.min(1, Math.max(0, v));
    }
    return out;
  }

  /* Gaussian low-pass of a sampled signal, evaluated only at the requested sample indices. */
  function filterAt(x, dt, bw, idx) {
    if (!bw) return idx.map(i => x[i]);
    const sigma = Math.sqrt(Math.LN2) / (2 * Math.PI * bw) / dt;
    const K = Math.ceil(5 * sigma);
    const w = new Float64Array(2 * K + 1);
    let tot = 0;
    for (let j = -K; j <= K; j++) { w[j + K] = Math.exp(-0.5 * (j / sigma) ** 2); tot += w[j + K]; }
    return idx.map(i => {
      let s = 0;
      for (let j = -K; j <= K; j++) { const ii = Math.min(x.length - 1, Math.max(0, i + j)); s += w[j + K] * x[ii]; }
      return s / tot;
    });
  }

  function gauss(rng) {                 // Box-Muller from a uniform generator
    let u = 0, v = 0;
    while (u === 0) u = rng(); v = rng();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  }

  /* The whole optical XOR box: two bit streams in, decided bits out.
     opts: rate (b/s), bwMod, bwRx (Hz or null), pLaser (W), tia (A/rtHz), rng, dt (s). */
  function xorStream(circuit, a, b, opts) {
    const rate = opts.rate, dt = opts.dt || Math.min(0.05e-12, 1 / rate / 100);
    const net = build(circuit, dt);
    const T = 1 / rate, nb = a.length;
    const lead = Math.round(40e-12 / dt);
    const lat = circuit.latency_ps * 1e-12;
    const n = lead + Math.ceil((nb * T + lat + 4 * T + 30e-12) / dt);
    const phases = Object.assign({}, circuit.base);
    for (const [key, bits] of [["A", a], ["B", b]]) {
      const s = circuit.inputs[key];
      const delta = ((s.on - s.off + Math.PI) % (2 * Math.PI) + 2 * Math.PI) % (2 * Math.PI) - Math.PI;
      const d = drive(bits, rate, dt, n, lead, opts.bwMod);
      const arr = new Float64Array(n);
      for (let i = 0; i < n; i++) arr[i] = s.off + delta * d[i];
      phases[s.role] = arr;
    }
    const e = run(net, n, phases, "xor");
    const pl = opts.pLaser || 10e-3;
    const cur = new Float64Array(n);
    for (let i = 0; i < n; i++) cur[i] = pl * (e.re[i] ** 2 + e.im[i] ** 2);      // photocurrent, A (1 A/W)
    const idx = [];
    for (let k = 0; k < nb; k++) idx.push(Math.min(n - 1, Math.round(lead + ((k + 0.5) * T + lat) / dt)));
    const smp = filterAt(cur, dt, opts.bwRx, idx);
    const one = pl * 0.5 * (circuit.levels_W_per_W["01"] + circuit.levels_W_per_W["10"]);
    const zero = pl * 0.5 * (circuit.levels_W_per_W["00"] + circuit.levels_W_per_W["11"]);
    const thr = 0.5 * (one + zero);
    const bwN = opts.bwRx || 0.75 * rate;
    const tia = opts.tia === undefined ? 20e-12 : opts.tia;
    const rng = opts.rng || Math.random;
    const q = 1.602176634e-19;
    const noisy = smp.map(i => i + gauss(rng) * Math.sqrt((2 * q * Math.max(i, 0) + tia * tia) * bwN));
    const out = noisy.map(i => (i > thr ? 1 : 0));
    // eye trace: the filtered current on a coarse grid, folded on two bit periods centred on the samples
    const per = Math.max(1, Math.round(T / dt / 48));
    const grid = [];
    for (let i = 0; i < n; i += per) grid.push(i);
    const fil = filterAt(cur, dt, opts.bwRx, grid);
    return { bits: out, samples: smp, threshold: thr, one, zero, trace: { t0: lead * dt + lat, dt: per * dt, y: fil, T }, latency_ps: circuit.latency_ps, dt, steps: n };
  }

  const api = { build, run, drive, filterAt, xorStream, erf };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.OptXOR = api;
})(this);
