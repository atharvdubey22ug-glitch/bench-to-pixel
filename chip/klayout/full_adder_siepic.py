#!/usr/bin/env python3
"""
Optical full adder (paper Fig. 15, hybrid) as a native KLayout / SiEPIC-Tools layout for the
SiEPIC EBeam process (Applied Nanotools). The drawing code is in builder.py (shared with the 4-bit
ALU in alu4_siepic.py). Outputs, in this folder (or the folder given as the first argument):
  EBeam_heaters_atharvdubey_FullAdder.gds / .oas   layout (the .oas is the fabrication export)
  EBeam_heaters_atharvdubey_FullAdder.lyrdb        SiEPIC functional verification report
  netlist_half1.sp, netlist_half2.sp                SiEPIC SPICE netlists of the two test circuits
  extracted_netlist.json                            SiEPIC net extraction, read by ../full_adder_sim.py
  build_report.json                                 waveguide lengths, heater/pad map, check counts

Run:  pip install klayout siepic siepic_ebeam_pdk ; python3 full_adder_siepic.py
Open the .gds/.oas in KLayout with SiEPIC-Tools and the EBeam technology to inspect or edit it.
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from builder import Builder, USER  # noqa: E402

OUT = sys.argv[1] if len(sys.argv) > 1 else HERE
TOP = f"EBeam_heaters_{USER}_FullAdder"

b = Builder(TOP)
b.full_adder(0, 0.0, 0.0, opt_name=lambda h: f"FullAdderHalf{h}", pad_name=lambda h, k: f"FullAdder_P{h}{k}",
             pitch=(4000.0, 4000.0), loopback=True, prefix=False)
r = b.finish(OUT)
for h in (1, 2):
    with open(os.path.join(OUT, f"netlist_half{h}.sp"), "w") as f:
        f.write(r["spice"][f"opt_in_TE_1550_device_{USER}_FullAdderHalf{h}"])
with open(os.path.join(OUT, "extracted_netlist.json"), "w") as f:
    json.dump(dict(components=r["ext"], gc_roles={k[1]: [v.x, v.y] for k, v in b.gc_roles.items()}), f, indent=1)
report = dict(top_cell=TOP, die_um=r["die"], siepic_verification_errors=r["n_err"], components=len(r["ext"]),
              waveguide_crossings=r["crossings"], dangling_optical_pins=r["dangling"],
              lengths_um={k: round(v, 3) for k, v in b.lengths.items()},
              pads={f"block{blk['block']}": blk["pads"] for blk in b.blocks},
              heater_pads=[hp for blk in b.blocks for hp in blk["heaters"]])
with open(os.path.join(OUT, "build_report.json"), "w") as f:
    json.dump(report, f, indent=1)
print("die", report["die_um"], "| SiEPIC verification errors:", r["n_err"], "| components", len(r["ext"]),
      "| crossings", len(r["crossings"]), "| dangling optical pins", len(r["dangling"]))
