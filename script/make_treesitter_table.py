#!/usr/bin/env python3
"""
make_treesitter_table.py
-------------------------
Generate img/treesitterComparisonTable.tex: a standalone comparison of
this paper's Earley, GLL, RNGLR, and BRNGLR implementations against
Tree-sitter, on two kinds of corpora:

1. The tokenized Java corpus (results/benchmark_tree_sitter_java.csv),
   broken down by size bucket. Tree-sitter and this paper's four
   parsers were all measured on the same files in this one CSV.

2. The Gamma_2/Gamma_3 stress grammars, pooled across all inputs
   rather than bucketed by size. Tree-sitter's numbers live in
   results/benchmark_tree_sitter_gamma{2,3}.csv (Tree-sitter only);
   this paper's four parsers' numbers live in the matching
   results/benchmark_rlc_gamma{2,3}.csv, already used by
   make_rq4_external_table.py. Both files share the same 300 input
   file names per grammar, so this is a same-input comparison.

Kept as its own small table (rather than columns in the RQ4 or
CYK/Valiant tables) because Tree-sitter data only exists for these
corpora; folding it into either 12-row table would leave the column
empty everywhere else.
"""

import csv
import math
import os
import statistics
from collections import defaultdict

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
OUTPUT_TEX = os.path.join(PROJECT_ROOT, "img", "treesitterComparisonTable.tex")

PARSERS = ["TreeSitter", "Leo", "GLL", "RNGLR", "BRNGLR"]
DISPLAY = {
    "TreeSitter": "Tree-sitter",
    "Leo": "Earley",
    "GLL": "GLL",
    "RNGLR": "RNGLR",
    "BRNGLR": "BRNGLR",
}
BUCKET_ORDER = ["XS", "S", "M", "L", "XL"]

STRESS_GRAMMARS = [
    {
        "label": r"$\Gamma_2$",
        "treesitter_file": "benchmark_tree_sitter_gamma2.csv",
        "other_file": "benchmark_rlc_gamma2.csv",
    },
    {
        "label": r"$\Gamma_3$",
        "treesitter_file": "benchmark_tree_sitter_gamma3.csv",
        "other_file": "benchmark_rlc_gamma3.csv",
    },
]


def ns_to_ms(value):
    return float(value) / 1_000_000.0


def fmt_num(value):
    if math.isinf(value):
        return r"$\infty$"
    if value < 1:
        return f"{value:.3f}"
    if value < 10:
        return f"{value:.2f}"
    if value < 100:
        return f"{value:.1f}"
    return f"{value:.0f}"


def split_num(s):
    if "." in s:
        i, f = s.split(".", 1)
        return i, f
    return s, ""


def best_worst(values):
    vals = [v for v in values.values() if v is not None]
    if not vals:
        return None, None
    bv, wv = min(vals), max(vals)
    if bv == wv:
        return None, None
    return bv, wv


def tex_cell(value, max_int, max_frac, is_best, is_worst):
    if value is None:
        return r"\multicolumn{1}{c|}{---}"
    if math.isinf(value):
        body = r"$\infty$"
    else:
        formatted = fmt_num(value)
        ip, fp = split_num(formatted)
        pad_left = max_int - len(ip)
        body = (r"\hphantom{" + "0" * pad_left + "}" if pad_left > 0 else "") + ip
        if max_frac > 0:
            if fp:
                pad_right = max_frac - len(fp)
                body += "." + fp
                if pad_right > 0:
                    body += r"\hphantom{" + "0" * pad_right + "}"
            else:
                body += r"\hphantom{.}" + r"\hphantom{" + "0" * max_frac + "}"
    if is_best:
        return r"\best{" + body + "}"
    if is_worst:
        return r"\worst{" + body + "}"
    return body


def read_rows(filename):
    with open(os.path.join(RESULTS_DIR, filename), newline="") as fh:
        return list(csv.DictReader(fh))


def median_by_parser(rows, parser_key):
    values = []
    for row in rows:
        if row["parser"] != parser_key:
            continue
        if row.get("status", "OK") == "OK":
            values.append(ns_to_ms(row["median_time_ns"]))
        else:
            values.append(math.inf)
    return statistics.median(values) if values else None


def summarize_java_buckets():
    rows = read_rows("benchmark_tree_sitter_java.csv")
    by_bucket = defaultdict(list)
    for row in rows:
        by_bucket[row["size_category"]].append(row)

    summaries = []
    for bucket in BUCKET_ORDER:
        bucket_rows = by_bucket.get(bucket, [])
        files = len({row["file"] for row in bucket_rows if row.get("file")})
        tokens = [int(float(row["token_count"])) for row in bucket_rows
                  if row.get("parser") == "TreeSitterJava"]
        median_tokens = int(statistics.median(tokens)) if tokens else None

        parser_summary = {}
        for parser in PARSERS:
            key = "TreeSitterJava" if parser == "TreeSitter" else parser
            parser_summary[parser] = median_by_parser(bucket_rows, key)

        summaries.append({
            "label": bucket,
            "files": files,
            "tokens": median_tokens,
            "parsers": parser_summary,
        })
    return summaries


def summarize_stress(grammar):
    ts_rows = read_rows(grammar["treesitter_file"])
    other_rows = read_rows(grammar["other_file"])

    files = len({row["file"] for row in ts_rows if row.get("file")})
    tokens = [int(float(row["token_count"])) for row in ts_rows]
    median_tokens = int(statistics.median(tokens)) if tokens else None

    parser_summary = {}
    for parser in PARSERS:
        if parser == "TreeSitter":
            parser_summary[parser] = median_by_parser(ts_rows, "TreeSitter")
        else:
            parser_summary[parser] = median_by_parser(other_rows, parser)

    return {
        "label": grammar["label"],
        "files": files,
        "tokens": median_tokens,
        "parsers": parser_summary,
    }


def write_tex(java_summaries, stress_summaries):
    all_summaries = java_summaries + stress_summaries

    displayed = []
    for summary in all_summaries:
        row_vals = {}
        for parser in PARSERS:
            raw = summary["parsers"][parser]
            if raw is None:
                row_vals[parser] = None
            elif math.isinf(raw):
                row_vals[parser] = raw
            else:
                row_vals[parser] = float(fmt_num(raw))
        displayed.append(row_vals)

    col_max_int = {p: 0 for p in PARSERS}
    col_max_frac = {p: 0 for p in PARSERS}
    for row_vals in displayed:
        for parser in PARSERS:
            v = row_vals[parser]
            if v is None or math.isinf(v):
                continue
            ip, fp = split_num(fmt_num(v))
            col_max_int[parser] = max(col_max_int[parser], len(ip))
            col_max_frac[parser] = max(col_max_frac[parser], len(fp))

    lines = [
        r"\small",
        r"% Generated by make_treesitter_table.py -- do not edit by hand.",
        r"% Caption and label live in the main .tex file.",
        r"\begin{tabular}{|l|r|r|r|r|r|r|r|}",
        r"  \hline",
        r"  \multirow{2}{*}{\textbf{Grammar}} & \multirow{2}{*}{\textbf{Files}}"
        r" & \multirow{2}{*}{\textbf{Tokens}}"
        r" & \multirow{2}{*}{\textbf{Tree-sitter}}"
        r" & \multicolumn{4}{c|}{\textbf{Generalised (\millisecond)}} \\",
        r"  & & & & \textbf{Earley} & \textbf{GLL}"
        r" & \textbf{RNGLR} & \textbf{BRNGLR} \\",
        r"  \hline\hline",
    ]

    def emit_rows(summaries, offset):
        for idx, summary in enumerate(summaries):
            row_vals = displayed[offset + idx]
            best_value, worst_value = best_worst(row_vals)

            cells = []
            for parser in PARSERS:
                value = row_vals[parser]
                is_best = best_value is not None and value == best_value
                is_worst = worst_value is not None and value == worst_value
                cells.append(
                    tex_cell(value, col_max_int[parser], col_max_frac[parser],
                             is_best, is_worst)
                )

            lines.append(
                f"  {summary['label']} & {summary['files']} & {summary['tokens']}"
                f" & " + " & ".join(cells) + r" \\"
            )

    emit_rows(java_summaries, 0)
    lines.append(r"  \hline")
    emit_rows(stress_summaries, len(java_summaries))

    lines += [r"  \hline", r"\end{tabular}"]

    os.makedirs(os.path.dirname(OUTPUT_TEX), exist_ok=True)
    with open(OUTPUT_TEX, "w") as fh:
        fh.write("\n".join(lines) + "\n")


def main():
    java_summaries = summarize_java_buckets()
    stress_summaries = [summarize_stress(g) for g in STRESS_GRAMMARS]
    write_tex(java_summaries, stress_summaries)
    print(f"LaTeX table written to: {OUTPUT_TEX}")
    for s in java_summaries + stress_summaries:
        print(s["label"], s["files"], s["tokens"],
              {DISPLAY[p]: s["parsers"][p] for p in PARSERS})


if __name__ == "__main__":
    main()
