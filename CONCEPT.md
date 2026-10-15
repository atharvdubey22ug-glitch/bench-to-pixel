# What this project shows

**In one paragraph.** Interference logic computes by propagation, not by switching. The XOR from our bench experiment, mapped onto a silicon photonic chip, adds about 5 ps and has no clock, so its speed is set by the modulators and detector at its edges, not by the logic. An electronic chip doing the same serial XOR must first convert the stream into wide parallel words, which costs it at least about 2 ns. The advantage of optical logic is therefore latency and per-gate bandwidth for data already in optical form. It is not a general replacement for CMOS: interference logic cannot be cascaded without returning to electronics.

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
5. **Two interactive pages.**
   - [`encrypt/`](encrypt/): one-time-pad encryption on the simulated optical XOR against a modelled electronic SerDes chip, with first-bit latency 19 ps against at least 2 ns.
   - [`alu/`](alu/): the 4-bit ALU computing ADD, SUB, AND, OR and XOR from each bit's simulated light levels, including what happens without per-bit calibration.

## Questions to expect

- **"Isn't a CMOS gate also a few picoseconds?"** Yes. The difference is not gate delay but that CMOS cannot operate serially at line rate, so it pays the deserialisation cost. Per-gate bandwidth is the real distinction.
- **"Is throughput higher?"** No. Both keep up with the line rate. The gain is latency, and simplicity: one coupler against an array of gates plus serialiser–deserialiser circuits.
- **"Does it scale to general computing?"** Not as digital logic. Interference gates lack gain and need a common phase reference, so multi-stage logic needs electronic regeneration. Linear operations, such as optical matrix–vector multiplication for neural networks, scale better, and that is where the field is moving.
- **"What is measured and what is simulated?"** The bench gates were measured. Everything on chip is simulated from the foundry's component models, the electronic latency is a lower-bound model, and nothing has been fabricated.

## The open question

Where in a system does the latency of optical–electronic conversion cost less than the deserialisation it replaces? And can interference logic be made cascadable without returning to electronics, for example through phase-regenerating stages or nonlinear elements?
