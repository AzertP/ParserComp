#!/usr/bin/env python3
"""
make_rq4_external_table.py
--------------------------
Generate img/rq4ExternalRuntimeTable.tex for the RQ4 external-validation
experiments. These experiments are intentionally kept separate from the
controlled 22-grammar benchmark tables.
"""

import csv
import math
import os
import statistics
from collections import OrderedDict, defaultdict

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
OUTPUT_TEX = os.path.join(PROJECT_ROOT, "img", "rq4ExternalRuntimeTable.tex")
OUTPUT_CSV = os.path.join(PROJECT_ROOT, "table_rq4_external_runtime.csv")

PARSERS = ["Leo", "GLL", "RNGLR", "BRNGLR"]
DISPLAY = {
    "Leo": "Earley",
    "GLL": "GLL",
    "RNGLR": "RNGLR",
    "BRNGLR": "BRNGLR",
}

EXPERIMENTS = [
    {
        "group": "CodeNet/Aizu",
        "corpus": "C",
        "mode": "Scannerless",
        "file": "benchmark_c_ws.csv",
    },
    {
        "group": "CodeNet/Aizu",
        "corpus": "C",
        "mode": "Tokenized",
        "file": "benchmark_c_tok.csv",
    },
    {
        "group": "CodeNet/Aizu",
        "corpus": "C++",
        "mode": "Scannerless",
        "file": "benchmark_cpp_ws.csv",
    },
    {
        "group": "CodeNet/Aizu",
        "corpus": "C++",
        "mode": "Tokenized",
        "file": "benchmark_cpp_tok.csv",
    },
    {
        "group": "CodeNet/Aizu",
        "corpus": "Java",
        "mode": "Scannerless",
        "file": "benchmark_java_ws.csv",
    },
    {
        "group": "CodeNet/Aizu",
        "corpus": "Java",
        "mode": "Tokenized",
        "file": "benchmark_java_tok.csv",
    },
    {
        "group": "CodeNet/Aizu",
        "corpus": "C\\#",
        "mode": "Tokenized",
        "file": "benchmark_csharp_tok.csv",
    },
    {
        "group": "Other",
        "corpus": "RLC Java",
        "mode": "Tokenized",
        "file": "benchmark_rlc_java.csv",
    },
    {
        "group": "Other",
        "corpus": "Pascal",
        "mode": "Tokenized",
        "file": "benchmark_pascal_tok.csv",
    },
    {
        "group": "Other",
        "corpus": "RLC SML",
        "mode": "Tokenized",
        "file": "benchmark_rlc_sml.csv",
    },
    {
        "group": "Stress",
        "corpus": r"$\Gamma_2$",
        "mode": "Tokenized",
        "file": "benchmark_rlc_gamma2.csv",
    },
    {
        "group": "Stress",
        "corpus": r"$\Gamma_3$",
        "mode": "Tokenized",
        "file": "benchmark_rlc_gamma3.csv",
    },
]


def read_rows(filename):
    with open(os.path.join(RESULTS_DIR, filename), newline="") as fh:
        return list(csv.DictReader(fh))


def input_key(row):
    return row.get("file") or row.get("source_file")


def ns_to_ms(value):
    return float(value) / 1_000_000.0


def median_token_count(rows):
    by_input = {}
    for row in rows:
        key = input_key(row)
        if not key:
            continue
        by_input[key] = int(float(row["token_count"]))
    return int(statistics.median(by_input.values()))


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
    """Split a plain formatted number like '15.3' or '356' into
    (integer_part, fractional_part). fractional_part is '' if the
    number has no decimal point."""
    if "." in s:
        i, f = s.split(".", 1)
        return i, f
    return s, ""


def best_worst(values):
    """Return (best_value, worst_value) among the present (non-None)
    displayed values for a row, or (None, None) if there is no visible
    distinction to highlight (no data present, or every present value
    is the same, including "every parser timed out")."""
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


def summarize_experiment(exp):
    rows = read_rows(exp["file"])
    total_inputs = len({input_key(row) for row in rows if input_key(row)})
    parser_rows = defaultdict(list)
    for row in rows:
        parser_rows[row["parser"]].append(row)

    parser_summary = {}
    for parser in PARSERS:
        rows_for_parser = parser_rows.get(parser, [])
        values = []
        for row in rows_for_parser:
            if row.get("status", "OK") == "OK":
                values.append(ns_to_ms(row["median_time_ns"]))
            else:
                values.append(math.inf)
        parser_summary[parser] = {
            "median_ms": statistics.median(values) if values else None,
        }

    return {
        "group": exp["group"],
        "corpus": exp["corpus"],
        "mode": exp["mode"],
        "file": exp["file"],
        "inputs": total_inputs,
        "tokens": median_token_count(rows),
        "parsers": parser_summary,
    }


def write_csv(summaries):
    fieldnames = ["group", "corpus", "mode", "inputs", "tokens"]
    for parser in PARSERS:
        name = DISPLAY[parser]
        fieldnames.append(f"{name}_median_ms")

    with open(OUTPUT_CSV, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for summary in summaries:
            row = {
                "group": summary["group"],
                "corpus": summary["corpus"],
                "mode": summary["mode"],
                "inputs": summary["inputs"],
                "tokens": summary["tokens"],
            }
            for parser in PARSERS:
                name = DISPLAY[parser]
                ps = summary["parsers"][parser]
                if ps["median_ms"] is None:
                    row[f"{name}_median_ms"] = ""
                elif math.isinf(ps["median_ms"]):
                    row[f"{name}_median_ms"] = "inf"
                else:
                    row[f"{name}_median_ms"] = f"{ps['median_ms']:.6f}"
            writer.writerow(row)


def write_tex(summaries):
    group_counts = OrderedDict()
    for summary in summaries:
        group_counts[summary["group"]] = group_counts.get(summary["group"], 0) + 1

    # Round every present, finite value to its displayed precision up
    # front: best/worst is judged on what's actually printed (matching
    # make_runtime_table.py / make_memory_table.py), not the unrounded
    # raw median, and infinities (timeouts) pass through unchanged.
    displayed = []
    for summary in summaries:
        row_vals = {}
        for parser in PARSERS:
            raw = summary["parsers"][parser]["median_ms"]
            if raw is None:
                row_vals[parser] = None
            elif math.isinf(raw):
                row_vals[parser] = raw
            else:
                row_vals[parser] = float(fmt_num(raw))
        displayed.append(row_vals)

    # Per-column max integer/fractional digit widths (finite values only),
    # so decimal points line up vertically within a column.
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
        r"% Generated by make_rq4_external_table.py -- do not edit by hand.",
        r"% Caption and label live in the main .tex file.",
        r"\begin{tabular}{|p{1.6em}|l|l|l|l|l|l|l|l|}",
        r"  \hline",
        r"  & \multirow{2}{*}{\textbf{Corpus}}"
        r" & \multirow{2}{*}{\textbf{Mode}}"
        r" & \multirow{2}{*}{\textbf{Files}}"
        r" & \multirow{2}{*}{\textbf{Tokens}}"
        r" & \multicolumn{4}{c|}{\textbf{Generalised (\millisecond)}} \\",
        r"  & & & & & \textbf{Earley} & \textbf{GLL}"
        r" & \textbf{RNGLR} & \textbf{BRNGLR} \\",
        r"  \hline\hline",
    ]

    previous_group = None
    for idx, summary in enumerate(summaries):
        group = summary["group"]
        group_cell = ""
        if group != previous_group:
            if previous_group is not None:
                lines.append(r"  \hline")
            group_cell = (
                rf"\multirow{{{group_counts[group]}}}{{*}}"
                rf"{{\rotatebox{{90}}{{\small {group}}}}}"
            )
            previous_group = group

        row_vals = displayed[idx]
        best_value, worst_value = best_worst(row_vals)

        parser_cells = []
        for parser in PARSERS:
            value = row_vals[parser]
            is_best = best_value is not None and value == best_value
            is_worst = worst_value is not None and value == worst_value
            parser_cells.append(
                tex_cell(
                    value, col_max_int[parser], col_max_frac[parser],
                    is_best, is_worst,
                )
            )

        lines.append(
            f"  {group_cell} & {summary['corpus']} & {summary['mode']}"
            f" & {summary['inputs']} & {summary['tokens']}"
            f" & " + " & ".join(parser_cells) + r" \\"
        )

    lines += [r"  \hline", r"\end{tabular}"]

    os.makedirs(os.path.dirname(OUTPUT_TEX), exist_ok=True)
    with open(OUTPUT_TEX, "w") as fh:
        fh.write("\n".join(lines) + "\n")


def main():
    summaries = [summarize_experiment(exp) for exp in EXPERIMENTS]
    write_csv(summaries)
    write_tex(summaries)
    print(f"CSV written to: {OUTPUT_CSV}")
    print(f"LaTeX table written to: {OUTPUT_TEX}")


if __name__ == "__main__":
    main()
