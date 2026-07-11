#!/usr/bin/env python3
"""
compute_lr_overhead.py
-----------------------
Compute the median/IQR/max runtime overhead of each generalized parser
relative to the LR(1) baseline, pooled across the five grammars that have
a genuine LR(1) variant: JSON (rr), JSON (lr), Expr (lr), Expr (rr), and
TinyC LR-1.

For each grammar and each matched token count present in both the LR(1)
CSV and a given generalized parser's CSV (status == OK on both sides), we
compute the ratio (generalized median_time_ns / LR(1) median_time_ns).
Ratios are pooled across all five grammars and all matched token counts,
then summarized by median, interquartile range, and maximum.

This script is not wired into the Makefile (the numbers it produces are
quoted directly in the RQ2/Discussion prose rather than a table), but it
is kept here so the statistic is reproducible from the raw results/*.csv
files rather than computed by hand.

Usage:
    python3 bin/compute_lr_overhead.py
"""

import csv
import os
import statistics

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

# Grammar label -> CSV file. These are the five grammars in the controlled
# benchmark that have a genuine LR(1)-refactored variant (as opposed to
# TinyPascal / S-Expr LL-1, which only have an LL(1) baseline).
FILES = {
    "JSON (rr)": "benchmark_json.csv",
    "JSON (lr)": "benchmark_json_lr.csv",
    "Expr (lr)": "benchmark_calc.csv",
    "Expr (rr)": "benchmark_expr.csv",
    "TinyC LR-1": "benchmark_tinyc_lr.csv",
}

PARSERS = ["Leo", "GLL", "RNGLR", "BRNGLR"]
DISPLAY = {"Leo": "Earley", "GLL": "GLL", "RNGLR": "RNGLR", "BRNGLR": "BRNGLR"}


def load(fname):
    with open(os.path.join(RESULTS_DIR, fname), newline="") as fh:
        return list(csv.DictReader(fh))


def main():
    overall = {p: [] for p in PARSERS}
    for label, fname in FILES.items():
        rows = load(fname)
        by_parser = {}
        for r in rows:
            if r.get("status", "OK") != "OK":
                continue
            by_parser.setdefault(r["parser"], {})[int(r["token_count"])] = \
                float(r["median_time_ns"])
        lr1 = by_parser.get("LR", {})
        if not lr1:
            print(f"{label}: no LR(1) data, skipping")
            continue
        for p in PARSERS:
            pd = by_parser.get(p, {})
            for tok, t in pd.items():
                if tok in lr1 and lr1[tok] > 0:
                    overall[p].append(t / lr1[tok])

    print(f"{'Parser':8s} {'n':>5s} {'median':>8s} {'IQR':>16s} {'max':>8s}")
    for p in PARSERS:
        vals = sorted(overall[p])
        if not vals:
            continue
        med = statistics.median(vals)
        q1 = statistics.quantiles(vals, n=4)[0]
        q3 = statistics.quantiles(vals, n=4)[2]
        mx = max(vals)
        print(f"{DISPLAY[p]:8s} {len(vals):5d} {med:8.2f} "
              f"[{q1:6.2f},{q3:6.2f}] {mx:8.2f}")


if __name__ == "__main__":
    main()
