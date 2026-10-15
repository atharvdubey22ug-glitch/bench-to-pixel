#!/usr/bin/env python3
"""Assemble the ALU page from src/head.html, src/body.html, alu.js, levels.json and layout.jpg.

  index.html                 full document, for GitHub Pages
  <outdir>/artifact.html     page content only (the Artifact host adds the document skeleton)
"""
import base64
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
head = open(os.path.join(HERE, "src", "head.html")).read()
body = open(os.path.join(HERE, "src", "body.html")).read()
alu = open(os.path.join(HERE, "alu.js")).read()
levels = json.load(open(os.path.join(HERE, "levels.json")))
res = json.load(open(os.path.join(HERE, "..", "chip", "alu4_results.json")))
levels["min_laser_total_uW"] = {k: res[k]["min_laser_total_uW"] for k in levels["dies"] if k in res}
img = "data:image/jpeg;base64," + base64.b64encode(open(os.path.join(HERE, "layout.jpg"), "rb").read()).decode()
lv = json.dumps(levels, separators=(",", ":"))
assert "</script" not in alu and "</script" not in lv
body = body.replace("/*LEVELS*/", lv).replace("/*ALUJS*/", alu).replace("/*LAYOUT*/", img)
full = ('<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
        '<link rel="icon" href="data:image/svg+xml,%3Csvg xmlns=\'http://www.w3.org/2000/svg\' viewBox=\'0 0 64 64\'%3E%3Crect width=\'64\' height=\'64\' rx=\'14\' fill=\'%23111a1e\'/%3E%3Cpath d=\'M8 40 L28 24 L56 24\' stroke=\'%23ff5a47\' stroke-width=\'4\' fill=\'none\' stroke-linecap=\'round\'/%3E%3Ccircle cx=\'28\' cy=\'24\' r=\'5\' fill=\'%23eba340\'/%3E%3C/svg%3E">\n'
        '<style>body{margin:0}img{max-width:100%}[hidden]{display:none!important}'
        ':root{padding-top:env(safe-area-inset-top,0px);padding-bottom:env(safe-area-inset-bottom,0px)}</style>\n'
        + head + "\n</head>\n<body>\n" + body + "\n</body>\n</html>\n")
open(os.path.join(HERE, "index.html"), "w").write(full)
out = sys.argv[1] if len(sys.argv) > 1 else None
if out:
    os.makedirs(out, exist_ok=True)
    open(os.path.join(out, "artifact.html"), "w").write(head + "\n" + body)
print("built", len(full), "bytes")
