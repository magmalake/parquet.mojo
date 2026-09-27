#!/usr/bin/env python3
"""Compare benchmark reports: median of the per-run medians, before vs after.

    python3 tools/bench_compare.py build/before-*.json -- build/after-*.json
"""

import json
import statistics
import sys


def medians(paths):
    out = {}
    for path in paths:
        with open(path) as f:
            for r in json.load(f)["results"]:
                out.setdefault(r["name"], []).append(r["median_ns"] / 1e6)
    return {name: statistics.median(v) for name, v in out.items()}


def main(argv):
    if "--" not in argv:
        sys.exit(__doc__.strip())
    split = argv.index("--")
    before, after = medians(argv[:split]), medians(argv[split + 1 :])
    width = max(len(n) for n in before)
    for name, b in before.items():
        if name in after:
            a = after[name]
            print(f"{name:{width}}  {b:9.3f} ms  {a:9.3f} ms  {(a / b - 1) * 100:+6.1f}%")


if __name__ == "__main__":
    main(sys.argv[1:])
