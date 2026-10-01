#!/usr/bin/env python3
"""
compute_lr_overhead.py
-----------------------
Compute the median/IQR/max runtime overhead of each generalized parser
relative to the LR(1) baseline, pooled across the five grammars that have
a genuine LR(1) variant: JSON (rr), JSON (lr), Expr (lr), Expr (rr), and
TinyC LR-1.

For each grammar, parser rows are paired with LR(1) rows in input order.
The driver writes every parser for an input before moving to the next input,
so this preserves distinct inputs that happen to have the same token count.
The script verifies matching input and token lengths before computing each
ratio. Both rows must be successful and pass the recorded correctness checks.
Ratios are pooled across all five grammars and all matched token counts,
then summarized by median, interquartile range, and maximum.

This script is not wired into the Makefile (the numbers it produces are
quoted directly in the RQ2/Discussion prose rather than a table), but it
is kept here so the statistic is reproducible from the raw results/benchmark_csv/*.csv
files rather than computed by hand.

Usage:
    python3 script/compute_lr_overhead.py
"""

import csv
import os
import statistics

from artifact_paths import RESULT_ROOT
RESULTS_DIR = os.path.join(RESULT_ROOT, "benchmark_csv")

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


def successful(row):
    return (
        row.get("status", "OK") == "OK"
        and row.get("recognized", "true") == "true"
        and row.get("parse_correct", "true") == "true"
    )


def main():
    overall = {p: [] for p in PARSERS}
    for label, fname in FILES.items():
        rows = load(fname)
        by_parser = {}
        for r in rows:
            by_parser.setdefault(r["parser"], []).append(r)
        lr1 = by_parser.get("LR", [])
        if not lr1:
            print(f"{label}: no LR(1) data, skipping")
            continue
        for p in PARSERS:
            parser_rows = by_parser.get(p, [])
            if len(parser_rows) != len(lr1):
                raise ValueError(
                    f"{label}: {p} has {len(parser_rows)} rows but LR has "
                    f"{len(lr1)}"
                )
            for index, (candidate, baseline) in enumerate(
                zip(parser_rows, lr1, strict=True)
            ):
                candidate_key = (
                    candidate["input_length"], candidate["token_count"]
                )
                baseline_key = (
                    baseline["input_length"], baseline["token_count"]
                )
                if candidate_key != baseline_key:
                    raise ValueError(
                        f"{label}: row {index} input mismatch for {p}: "
                        f"{candidate_key} != {baseline_key}"
                    )
                if successful(candidate) and successful(baseline):
                    baseline_time = float(baseline["median_time_ns"])
                    if baseline_time > 0:
                        overall[p].append(
                            float(candidate["median_time_ns"]) / baseline_time
                        )

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
