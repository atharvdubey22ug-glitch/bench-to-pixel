#!/usr/bin/env python3
"""
Run the EBeam PDK's own KLayout DRC decks (siepic_ebeam_pdk/drc/*.lydrc) on a layout, in batch mode
with the KLayout DRC engine (needs the `klayout` executable, e.g. `apt install klayout`).

  python3 run_drc.py <layout.gds|.oas> [outdir]

Writes <outdir>/drc/<deck>.lyrdb (open in KLayout's Marker Browser) and <outdir>/drc/summary.json.
Decks:
  SiEPIC_EBeam_DRC       the process DRC (Si, SiN, M1 heater, M2 router, M_Open, floor plan) -- the one
                         that applies to this design
  ELEC463 Metal Heater   course deck "Chip1": checks metal on layer 47/0, which this design does not use
  ELEC463 Doping         course deck "Chip2": a doped-heater process that forbids metal over silicon,
                         so every heater of this design is flagged (M_Si_separation); not applicable
"""
import html
import json
import os
import re
import subprocess
import sys
from collections import Counter

import siepic_ebeam_pdk

DECKS = os.path.join(os.path.dirname(siepic_ebeam_pdk.__file__), "drc")


def run(layout, outdir):
    os.makedirs(os.path.join(outdir, "drc"), exist_ok=True)
    res = {}
    for f in sorted(os.listdir(DECKS)):
        if not f.endswith(".lydrc"):
            continue
        text = html.unescape(re.search(r"<text>(.*)</text>", open(os.path.join(DECKS, f)).read(), re.S).group(1))
        text = re.sub(r'^report\((".*?")\)', r'source($input)\nreport(\1, $output)', text, count=1, flags=re.M)
        name = f[:-6]
        script = os.path.join(outdir, "drc", name.replace(" ", "_") + ".drc")
        rdb = os.path.abspath(os.path.join(outdir, "drc", name.replace(" ", "_") + ".lyrdb"))
        open(script, "w").write(text)
        subprocess.run(["klayout", "-b", "-r", script, "-rd", f"input={os.path.abspath(layout)}", "-rd", f"output={rdb}"],
                       check=True, capture_output=True)
        os.remove(script)
        x = open(rdb).read()
        rules = re.findall(r"<category>\s*<name>(.*?)</name>", x, re.S)
        hits = Counter(re.findall(r"<item>.*?<category>(.*?)</category>", x, re.S))
        res[name] = dict(rules_checked=len(rules), violations=sum(hits.values()), by_rule=dict(hits))
    return res


if __name__ == "__main__":
    lay = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else os.path.dirname(os.path.abspath(lay))
    r = run(lay, out)
    with open(os.path.join(out, "drc", "summary.json"), "w") as f:
        json.dump(r, f, indent=1)
    for k, v in r.items():
        print(f"{k:22s} rules {v['rules_checked']:3d}  violations {v['violations']:4d}  {v['by_rule'] or ''}")
