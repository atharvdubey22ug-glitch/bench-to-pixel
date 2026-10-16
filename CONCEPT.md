# What this project shows

**In one paragraph.** Interference logic computes by propagation, not by switching. The XOR from our bench experiment, mapped onto a silicon photonic chip, adds about 5 ps and has no clock, so its speed is set by the modulators and detector at its edges, not by the logic. An electronic chip doing the same serial XOR must first convert the stream into wide parallel words, which costs it at least about 2 ns. The advantage of optical logic is therefore latency and per-gate bandwidth for data already in optical form. It is not a general replacement for CMOS. Gates that switch beams on and off cannot be cascaded without returning to electronics. Encoding bits as phase lets XORs chain (eight in one pass on the parity chip), but their errors add up too, because nothing restores the signal.

## The concept: computation by propagation

- **CMOS.** A result exists only after transistors charge and discharge. Each operation has a fixed switching time, so logic is clocked at a few GHz. A 100 Gb/s serial stream must be deserialised into many parallel lanes before a gate can act on it, and reserialised afterwards. That conversion, not the gate delay, dominates latency.
- **Interference.** The result is the superposition of two fields. Nothing switches, so there is no clock and no settling time. The output exists as soon as light has crossed the device, and one gate processes the stream serially at whatever rate the inputs can be imposed on light.

The trade-off: optics offers time-of-flight latency and very large bandwidth per gate. It lacks gain and signal restoration, so optical gates cannot be chained indefinitely.

## What was done

1. **Bench experiment** (the paper). Interference-based AND, OR and XOR gates, and half and full adders, in free-space optics with a polarised laser, beamsplitters, polarisers and a Michelson interferometer.
2. **Foundry layouts.** The full adder, then a 4-bit ripple-carry ALU built from four of them, drawn in KLayout for the SiEPIC EBeam silicon photonics process. Both pass SiEPIC functional verification and the process design-rule check (`chip/`).
3. **Two findings from the chip design.**
   - **A light-only cascade fails.** XOR 1 leaves "only A" and "only B" with opposite optical phase, so no carry-in beam can cancel both. Multi-stage logic therefore needs an optical–electronic–optical step between stages: the cascadability problem of optical logic, met in practice. The ALU's carry chain is this hybrid step.
   - **The Michelson geometry limits speed.** Each input is crossed twice, about 19 ps apart, which caps the paper's Michelson at 25 Gb/s. A single-pass variant removes the limit.
4. **Time-domain simulation from the layouts** (`chip/speed/`). It agrees with the frequency-domain model to numerical precision. The single-pass XOR computes on two independent bit streams at 200 Gb/s with 100 GHz modulators, within 0.8 dB of a plain waveguide carrying no logic. Its on/off ratio stays above 30 dB over about 2.5 THz of optical bandwidth.
5. **A cascade that works: phase-encoded parity** (`chip/README.md#phase-encoded-parity`). Each bit is a phase, 0 or π, on one beam, so the light leaving one section is a valid input for the next. Eight sections in one arm compute the parity of eight bits in one pass, with no detection in between.
   - Every one of the 256 inputs is correct on 180 simulated dies across all process corners, calibrated from the grating-coupler readings alone.
   - With aligned drives it runs at 150 Gb/s with 100 GHz modulators.
   - The cost is that phase errors add along the chain, as the bits do. Each drive must be accurate to about 1–2%, because nothing restores the signal.
6. **A neuron made of light** (`chip/README.md#optical-neuron`). The parity chip's phase sections are put side by side instead of in a row. Sixteen arms each carry one input as a phase, 0 or π, set by whether the input agrees with its weight. A combiner tree adds the sixteen fields in one pass, and balanced detection against a reference reads the signed sum. That is a binary-weight neuron (the XNOR-and-count unit of binarised neural networks), and the optical version of the bench's carry gate, where polarisers set the weights.
   - With 3 × 5 pixel digits stored as weights, the chip picks the same digit as an ideal neuron on all 180 simulated dies across the process corners.
   - It needs a three-step calibration from the output readings alone (phases, a heater-crosstalk map the driver cancels, a balance weight on one photodiode). With it every output level stays distinct on all 180 dies; without it they merge on some dies.
   - With aligned drives and 100 GHz modulators, one neuron takes a whole image per bit slot: every output level stays distinct to 100 G images/s, and its yes/no decision holds to 125.
7. **Four interactive pages.**
   - [`encrypt/`](encrypt/): one-time-pad encryption on the simulated optical XOR against a modelled electronic SerDes chip, with first-bit latency 19 ps against at least 2 ns.
   - [`alu/`](alu/): the 4-bit ALU computing ADD, SUB, AND, OR and XOR from each bit's simulated light levels, including what happens without per-bit calibration.
   - [`parity/`](parity/): the parity chip live, the cost of drive errors along the chain, and its eye diagrams at up to 400 Gb/s.
   - [`perceptron/`](perceptron/): draw a digit and read the neuron chip's ten scores, how exact its sum is, and its eye diagrams.

## Questions to expect

- **"Isn't a CMOS gate also a few picoseconds?"** Yes. The difference is not gate delay but that CMOS cannot operate serially at line rate, so it pays the deserialisation cost. Per-gate bandwidth is the real distinction.
- **"Is throughput higher?"** No. Both keep up with the line rate. The gain is latency, and simplicity: one coupler against an array of gates plus serialiser–deserialiser circuits.
- **"Does it scale to general computing?"** Not as digital logic. Interference gates lack gain and need a common phase reference. Phase encoding lets XORs chain (the parity chip), but errors accumulate without regeneration, and AND/OR still need a nonlinearity or electronics. Linear operations scale better: the neuron chip adds sixteen weighted inputs in one pass, and the field is moving toward optical matrix–vector multiplication for neural networks. Linear operations, such as optical matrix–vector multiplication for neural networks, scale better, and that is where the field is moving.
- **"What is measured and what is simulated?"** The bench gates were measured. Everything on chip is simulated from the foundry's component models, the electronic latency is a lower-bound model, and nothing has been fabricated.

## The open question

Where in a system does the latency of optical–electronic conversion cost less than the deserialisation it replaces? Phase encoding makes XOR chains cascadable without returning to electronics (the parity chip). Can the errors that accumulate along them be removed optically, for example with phase-sensitive amplifiers that restore 0 and π, and can AND and OR be added without detection?
