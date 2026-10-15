/* The 4-bit optical ALU, driven by the simulated light levels in levels.json (chip/alu4_export_web.py).
   Each bit: inputs set its heaters; XOR 1 is read and thresholded (X); X is re-applied with the
   carry-in; SUM and CARRY are read and thresholded; the carry comparator drives the next bit's
   carry-in heaters. Same procedure as chip/alu4_sim.py. */
(function (root) {
  "use strict";
  const Q = 1.602176634e-19;

  function sigma(i, det) {                       // detector noise (A): shot + amplifier + laser RIN
    const v = 2 * Q * (Math.max(i, 0) + det.dark) + (det.tia_pA * 1e-12) ** 2 + i * i * Math.pow(10, det.rin_dB / 10);
    return Math.sqrt(v * det.bw);
  }

  function gauss(rng) {
    let u = 0; while (u === 0) u = rng();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * rng());
  }

  /* One bit. pBit: laser power per bit (W). rng: null for no noise. Returns decisions and currents. */
  function bit(b, a, bb, c, pBit, det, rng) {
    const idx = x => 8 * a + 4 * bb + 2 * c + x;
    const read = (v, thr) => {
      const i = det.resp * pBit * v;
      const n = rng ? gauss(rng) * sigma(i, det) : 0;
      return { i: i + n, ideal: i, thr: det.resp * pBit * thr, sig: sigma(i, det) };
    };
    const x1 = read(b.xor1[idx(0)], b.thr.xor1);
    const X = x1.i > x1.thr ? 1 : 0;
    const s = read(b.sum[idx(X)], b.thr.sum);
    const k = read(b.carry[idx(X)], b.thr.carry);
    return { a, b: bb, c, X, xor1: x1, sum: s, carry: k, S: s.i > s.thr ? 1 : 0, C: k.i > k.thr ? 1 : 0 };
  }

  /* op: ADD | SUB | AND | OR | XOR. Returns per-bit detail, result, carry out and the reference. */
  function run(die, det, op, A, B, cin, pBit, rng) {
    const bits = [];
    let c = op === "SUB" ? 1 : (op === "ADD" ? cin : 0);
    let res = 0;
    for (let i = 0; i < 4; i++) {
      const a = (A >> i) & 1;
      let bb = (B >> i) & 1;
      if (op === "SUB") bb = 1 - bb;
      if (op === "AND") c = 0;
      if (op === "OR") c = 1;
      const r = bit(die.bits[i], a, bb, c, pBit, det, rng);
      r.cin = c;
      const out = op === "XOR" ? r.X : (op === "AND" || op === "OR") ? r.C : r.S;
      r.out = out;
      res |= out << i;
      bits.push(r);
      c = r.C;
    }
    const cout = (op === "ADD" || op === "SUB") ? c : null;
    return { bits, result: res, cout, ref: reference(op, A, B, cin) };
  }

  function reference(op, A, B, cin) {
    if (op === "ADD") { const s = A + B + cin; return { result: s & 15, cout: s >> 4 }; }
    if (op === "SUB") { const s = A + (15 - B) + 1; return { result: s & 15, cout: s >> 4 }; }
    return { result: op === "AND" ? (A & B) : op === "OR" ? (A | B) : (A ^ B), cout: null };
  }

  /* Every operand pair of every operation (ADD with both carry-ins): 1,536 operations. */
  function exhaustive(die, det, pBit, rng) {
    const tally = {};
    let wrong = 0, total = 0;
    for (const op of ["ADD", "SUB", "AND", "OR", "XOR"]) {
      tally[op] = { wrong: 0, total: 0 };
      for (const cin of op === "ADD" ? [0, 1] : [0]) {
        for (let A = 0; A < 16; A++) for (let B = 0; B < 16; B++) {
          const r = run(die, det, op, A, B, cin, pBit, rng);
          const bad = r.result !== r.ref.result || (r.ref.cout !== null && r.cout !== r.ref.cout);
          tally[op].total++; total++;
          if (bad) { tally[op].wrong++; wrong++; }
        }
      }
    }
    return { tally, wrong, total };
  }

  const api = { run, exhaustive, reference, sigma };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.OptALU = api;
})(this);
