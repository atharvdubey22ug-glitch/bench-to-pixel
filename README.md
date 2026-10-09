# Bench to Pixel

An interactive Three.js page that starts on the optical half adder from our paper and zooms out, step by step, until the same circuit is part of a GPU drawing a frame on a monitor.

Open `index.html` in a browser. Scroll to zoom out; drag to look around.

## What the page shows

1. **The bench.** The half adder from *Using Polarisation and Interference for Optical Computing* (Figure 14, section 4.1): a linearly polarised laser, beamsplitters S1 (5), S2 (6) and S3 (7), polarisers (3, 4) for intensity marking, mirrors (1, 2) converging both beams on the carry photoresistor, and a Michelson interferometer with mirrors (9, 10), two lenses (8) and a screen (11) with a hole on a dark fringe. Inputs A and B can be toggled.
2. **The silicon cell.** The same layout as a silicon photonic circuit: MMI splitters, micro-ring modulators as inputs, variable attenuators as weights, loop mirrors, germanium photodiodes and a TIA + comparator.
3. **Optical tiles.** The carry gate read as a dot product, widened to 4×4 matrix–vector tiles.
4. **Shader cluster and die.** An ATTILA-style unified-shader GPU with the tile blocks inside its shader clusters.
5. **Card and desk.** A 2008-era graphics card with a laser module and fibre, on a test bench, driving a 22-inch monitor over DVI. The teapot scene on the monitor is rendered live.

## What is real

- **Built (2024):** optical AND, OR and XOR gates, a half adder that was correct for every input, and a full adder limited by photoresistor sensitivity, misalignment and bulk.
- **Standard parts:** MMI splitters, ring modulators, attenuators, loop mirrors and germanium photodiodes are offered by silicon photonics foundries.
- **Proposed:** the 4×4 optical tile and its place in the GPU. It has not been simulated or fabricated.

Bench distances, the carry meter values and the breadboard size are approximations; the paper does not give them.

## Credits

- Paper: *Using Polarisation and Interference for Optical Computing*, Atharv Dubey, Sanya Davalbhakta and Proteep Mallik, Student Research Journal 2024, Azim Premji University.
- GPU model: ATTILA by V. Moya, C. González, J. Roca, A. Fernández and R. Espasa, Universitat Politècnica de Catalunya ([paper, ISPASS 2006](https://hgpu.org/?p=2152), [source mirror](https://github.com/cooperyuan/attila)).
- Built with [three.js](https://threejs.org) r128, loaded from cdnjs and jsDelivr.

[F²R · Fiction to Reality](https://f2r.site)
