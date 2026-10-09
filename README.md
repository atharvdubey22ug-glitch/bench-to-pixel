# Bench to Pixel

An interactive Three.js page that starts on the optical half adder from our paper and zooms out, step by step, until the same circuit is part of a GPU drawing a frame on a monitor.

Open `index.html` in a browser. Scroll to zoom out; drag to look around.

## What the page shows

1. **The bench.** The half adder from *Using Polarisation and Interference for Optical Computing* (Figure 14, section 4.1): a linearly polarised laser, beamsplitters S1 (5), S2 (6) and S3 (7), polarisers (3, 4) for intensity marking, mirrors (1, 2) converging both beams on the carry photoresistor, and a Michelson interferometer with mirrors (9, 10), two lenses (8) and a screen (11) with a hole on a dark fringe. Inputs A and B can be toggled.
2. **The silicon cell.** The same layout as a silicon photonic circuit: MMI splitters, micro-ring modulators as inputs, variable attenuators as weights, loop mirrors, germanium photodiodes and a TIA + comparator.
3. **Optical tiles.** The carry gate read as a dot product, widened to 4×4 matrix–vector tiles.
4. **Shader cluster and die.** An ATTILA-style unified-shader GPU with the tile blocks inside its shader clusters.
5. **Card and desk.** A 2008-era graphics card with a laser module and fibre, on a test bench, driving a 22-inch monitor over DVI. The teapot scene on the monitor is rendered live, with the right half lit through the simulated optical tile; sliders set laser power, ADC bits and ring temperature error.

## Simulation

`sim/optical_sim.py` is a circuit-level simulation: each component is an analytical transfer function (MMI splitters, all-pass ring modulators, tunable couplers, loop mirrors, germanium photodiodes) with shot, amplifier and laser intensity noise. It has no full-wave electromagnetic simulation, and every parameter at the top of the file is an assumption typical of a silicon photonics process at 1550 nm. `python3 sim/optical_sim.py` rewrites `sim/results.json` and the figures in about 7 seconds.

**Half-adder cell** (1 mW into the cell, 1 Gb/s inputs):

| A B | Carry (µA) | Sum (µA) |
|---|---|---|
| 0 0 | 2.5 | 0.02 |
| 0 1 | 89.8 | 8.06 |
| 1 0 | 60.7 | 8.24 |
| 1 1 | 148.0 | 0.02 |

- Carry decision Q ≈ 56 and sum decision Q ≈ 9; Q = 7 is one error in 10¹².
- The XOR uses the most light: below about 0.78 mW its error rate rises past 10⁻¹².
- The XOR's dark output holds within ±19° of bias phase, about 2.2 K of temperature difference between 100 µm arms.

![XOR bias-phase tolerance](sim/figures/xor_phase_tolerance.png)

**4×4 tile** (four WDM channels, 6-bit input DACs, 8-bit heater weights): 5.6 effective bits at 0.5 mW per channel and 5 GS/s, noise-limited; it saturates near 7 bits where laser intensity noise and the input DACs take over.

![Tile effective bits](sim/figures/tile_effective_bits.png)

**Lighting on the monitor.** The page lights the right half of the monitor through a per-pixel model of the tile (input DAC, ring drift, weight error, noise, ADC) fitted to `lighting_circuit()` in the simulation. At 0.5 mW the full circuit model gives an N·L error of 0.063 RMS and the page's model gives 0.062; with 0.1 K of ring drift, 0.188 and 0.189.

## What is real

- **Built (2024):** optical AND, OR and XOR gates, a half adder that was correct for every input, and a full adder limited by photoresistor sensitivity, misalignment and bulk.
- **Standard parts:** MMI splitters, ring modulators, attenuators, loop mirrors and germanium photodiodes are offered by silicon photonics foundries.
- **Simulated:** the half-adder cell and a 4×4 tile at circuit level, as above.
- **Proposed:** the tile's place inside the GPU. Nothing has been fabricated or simulated at the electromagnetic level, and every simulated number depends on the assumed parameters.

Bench distances, the carry meter values and the breadboard size are approximations; the paper does not give them.

## Credits

- Paper: *Using Polarisation and Interference for Optical Computing*, Atharv Dubey, Sanya Davalbhakta and Proteep Mallik, Student Research Journal 2024, Azim Premji University.
- GPU model: ATTILA by V. Moya, C. González, J. Roca, A. Fernández and R. Espasa, Universitat Politècnica de Catalunya ([paper, ISPASS 2006](https://hgpu.org/?p=2152), [source mirror](https://github.com/cooperyuan/attila)).
- Built with [three.js](https://threejs.org) r128, loaded from cdnjs and jsDelivr.

[F²R · Fiction to Reality](https://f2r.site)
