"""
run_all.py -- Master runner for the validation suite of
"The Directional Pair: A Model Without Identity, a Catalogue Without
Process, and the Single Table They Jointly Evaluate".

Runs every experiment, writes one JSON record per experiment to results/,
and writes an aggregate master record.

    python run_all.py

Determinism: every experiment seeds its own RNG from a fixed base seed, so
the suite reproduces identically across runs and platforms.
"""

from __future__ import annotations

import json
import os
import platform
import sys
import time

import exp_directional
import exp_foundations
import exp_identity
import exp_probing
import exp_separation
from core import RESULTS_DIR, save

PARTS = [
    ("I -- Foundations", exp_foundations),
    ("II -- Identity", exp_identity),
    ("III -- The Directional Pair", exp_directional),
    ("IV -- Construction and Probing", exp_probing),
    ("V -- Closure and the Separation", exp_separation),
]


def main() -> int:
    t0 = time.time()
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("=" * 72)
    print("  THE DIRECTIONAL PAIR -- validation suite")
    print("=" * 72)

    summaries, all_rows = [], []
    for label, mod in PARTS:
        print(f"\n[{label}]")
        s = mod.run_all()
        summaries.append(s)
        for e in s["experiments"]:
            mark = "PASS" if e["verdict"] == "PASS" else "FAIL"
            print(f"  {e['id']}  {mark}")
            all_rows.append({"part": label, **e})

    elapsed = time.time() - t0
    n_total = len(all_rows)
    n_pass = sum(r["verdict"] == "PASS" for r in all_rows)

    master = {
        "title": "The Directional Pair -- validation suite",
        "n_experiments": n_total,
        "n_pass": n_pass,
        "n_fail": n_total - n_pass,
        "pass_rate": f"{n_pass}/{n_total}",
        "elapsed_seconds": round(elapsed, 3),
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
        },
        "determinism": "Every experiment seeds its own RNG from base seed 42; "
                       "the suite reproduces identically across runs.",
        "backend": "Exact Edmonds-Karp max-flow / min-cut, standard library "
                   "only; no external numerical dependency.",
        "parts": [
            {"part": s["part"], "pass_rate": s["pass_rate"]} for s in summaries
        ],
        "experiments": all_rows,
    }
    save("master_results", master)

    print("\n" + "=" * 72)
    for s in summaries:
        print(f"  {s['part']:<34s} {s['pass_rate']}")
    print("-" * 72)
    print(f"  {'AGGREGATE':<34s} {n_pass}/{n_total}"
          f"   ({elapsed:.1f}s)")
    print("=" * 72)
    print(f"\nResults written to {RESULTS_DIR}")

    return 0 if n_pass == n_total else 1


if __name__ == "__main__":
    raise SystemExit(main())
