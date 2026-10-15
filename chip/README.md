# Optical full adder chip

A silicon photonic version of the full adder in *Using Polarisation and Interference for Optical Computing* (Section 4.2, Figures 15–16), designed for the SiEPIC EBeam process run by Applied Nanotools: 220 nm silicon waveguides, TiW heaters (M1) and a metal routing layer (M2).

![Chip layout](figures/chip_layout.png)

`full_adder_chip.gds` is the layout: 1.95 × 1.36 mm, nine grating couplers in one fibre-array column at 127 µm pitch, and twelve probe pads. It has not been fabricated.

## From the bench to the chip

| Bench (Fig. 15/16) | Chip |
|---|---|
| Laser | External 1550 nm laser into a TE grating coupler |
| S1 (1): carry-in light | Y-branch S1: light for block 2 (carry-in half) and block 1 (A, B half) |
| S2 (2), S3 (3): inputs A, B for AND gate 1 | Y-branch taps Sa, Sb in block 1 |
| S4 (8) with mirrors 7, 9: XOR gate 1 | Michelson interferometer: broadband directional coupler, Sagnac loop mirrors |
| S5 (10), S6 (13) with mirrors 11, 12: AND gate 2, XOR gate 2 | Block 2, the same circuit as block 1 |
| Blocking a beam to set an input | Thermo-optic Mach–Zehnder switch in that branch or interferometer arm |
| Polarisers marking the intensities | The same switches, set to partial transmission |
| One "OR" photoresistor for the carry | Four carry outputs on photodiodes wired in parallel, so their currents add |
| Lenses, screen and hole on a dark fringe | Interferometer output port biased dark with a heater |
| Arduino threshold | Comparators |

### One change from Figure 15

The figure sends XOR 1's light into XOR 2. That cannot work with light alone. XOR 1 is dark when A and B are both open because their light arrives with opposite phase. So "only A" and "only B" leave XOR 1 with opposite phase (172° apart in the simulation). When that light is combined with the carry-in beam, one of the two cases cancels and the other doubles. No carry-in phase or power fixes both (below).

On the chip, XOR 1 is therefore read by a photodiode and comparator. Its result, X, drives the X switches of block 2, in the same way the bench's Arduino applied its threshold. Everything else follows the paper's arrangement.

![Cascade failure](figures/cascade_failure.png)

## Simulation

`full_adder_sim.py` simulates the whole chip as one scattering network in SAX. It uses the SiEPIC EBeam PDK compact models (grating couplers, Y-branches, broadband couplers and waveguides, via simphony 0.7.3) plus heater phase shifters, so all back-reflections are included.

Waveguide lengths are taken from the drawn layout (`layout_extract.json`), and the two Michelson arms are equalised in length. Heaters are calibrated per chip with sweeps that only read grating-coupler outputs, as on a probe station. Detection is off-chip: InGaAs photodiodes (1 A/W), 2 pA/√Hz amplifier noise, 1 MHz bandwidth and −140 dB/Hz laser intensity noise.

**All eight inputs, 1 mW of laser light in the fibre** (photocurrents in µA):

| A B Cin | X (XOR 1) | XOR 1 | SUM | CARRY |
|---|---|---|---|---|
| 0 0 0 | 0 | 0.12 | 0.12 | 0.0 |
| 0 0 1 | 0 | 0.12 | 7.35 | 32.9 |
| 0 1 0 | 1 | 6.79 | 6.85 | 34.2 |
| 0 1 1 | 1 | 6.79 | 0.00 | 66.9 |
| 1 0 0 | 1 | 7.35 | 6.85 | 32.9 |
| 1 0 1 | 1 | 7.35 | 0.00 | 65.7 |
| 1 1 0 | 0 | 0.00 | 0.12 | 69.4 |
| 1 1 1 | 0 | 0.00 | 7.34 | 102.2 |

![Truth table](figures/truth_table.png)

**Decision margins**, with thresholds set midway between the highest 0 and the lowest 1:

| Output | Contrast | Margin | Laser power drift tolerated |
|---|---|---|---|
| XOR 1 | 17.5 dB | 98% | ±49% |
| SUM | 17.5 dB | 98% | ±49% |
| CARRY | 2.8 dB | 48% | ±24% |

- **Laser power:** every output reaches Q ≥ 7 (one error in 10¹²) from 4.2 µW of laser power in the fibre. Optical noise is not the limit; leakage through the switches is.
- **Bias tolerance:** both Michelson biases tolerate ±60° of error.
- **Process corners:** at all nine PDK thickness (210/220/230 nm) and width (480/500/520 nm) corners, all three outputs are correct, with carry margins of 47–49%. This holds even with the nominal chip's heater settings, because the Michelson arms are equal in length.

**Carry marking.** The paper's AND-OR marking (Fig. 5) works. With the AND gate 2 branches each weighted half of the AND gate 1 branches, the carry margin is 24%.

The simulation also finds a better marking. With equal weights on A, B and Cin, and the X branch off, the carry detector computes majority(A, B, Cin), which equals the full-adder carry. The margin doubles to 48%, and the carry no longer waits for the XOR 1 comparator. The X branch switch stays on the chip, so both markings can be tested.

## Layout

`layout.py` draws the chip with gdsfactory using the official SiEPIC EBeam cells (`ebeam_gc_te1550`, `ebeam_y_1550`, `ebeam_bdc_te1550`, `ebeam_terminator_te1550`, from the `siepic_ebeam_pdk` package).

- **Blocks:** two identical half blocks, the second mirrored below the laser row.
- **Switches:** each is two broadband couplers with arms spread 40 µm apart and a 120 µm TiW heater on one arm. Unused ports are terminated.
- **Heater wiring:** five heaters per block run from six pads. Adjacent heaters share a pad, which keeps the single metal layer planar. Each heater sees the voltage difference between its two pads (`pad_map.json`).
- **Labels:** SiEPIC automated-measurement labels are on the laser and loopback couplers and on every pad.

`verify.py` checks the GDS. Everything passes:

- the SiEPIC EBeam DRC deck's rules: Si width ≥ 60 nm and space ≥ 70 nm; M1 width and space ≥ 3 µm; M2 width ≥ 5 µm and space ≥ 8 µm; pad openings ≥ 10 µm; M2-over-M1 overlap ≥ 3 µm; all silicon inside the floor plan;
- 12 metal nets, each with exactly one pad;
- each of the 10 heaters on exactly two nets;
- no metal within 25 µm of a grating coupler;
- no waveguide touching other silicon except at its own end ports.

The rules are reimplemented in Python with the KLayout API. The foundry's own KLayout DRC and SiEPIC verification should still be run before submission.

## Testing a fabricated chip

1. Align the fibre array on the loopback pair.
2. Laser on the LASER coupler at 1550 nm, about 1 mW in the fibre.
3. Calibrate the heaters in this order:
   - each AND switch, sweeping it on its own output;
   - each Michelson's arm switches, with a 2D sweep at a bright bias;
   - the biases, both arms on, minimising the output;
   - the carry marking.

   `Chip.calibrate()` is the reference procedure.
4. Apply A and B. Read XOR 1, threshold it, and apply the result to X along with Cin. Read SUM, and read CARRY as the sum of the four carry photocurrents.

## Limits

- No full-wave electromagnetic simulation beyond what the PDK models contain.
- Heater thermal crosstalk is not modelled; heaters are at least 90 µm apart.
- Heater power and switching speed depend on the TiW process and must be measured. Thermal switches settle in tens of microseconds, so this chip shows the logic, not speed.
- Waveguide loss (3 dB/cm) is an assumption.
- The die area and floor plan must match the run's submission template.

## Reproduce

```
pip install -r requirements.txt
python3 layout.py          # GDS, layout_extract.json, pad_map.json; checks waveguide crossings
python3 verify.py          # design-rule and electrical checks
python3 full_adder_sim.py  # about 20 minutes; full_adder_results.json
python3 figures.py
python3 render_chip.py
```
