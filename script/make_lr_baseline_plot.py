#!/usr/bin/env python3
"""
make_lr_baseline_plot.py
------------------------
Generate img/lrBaselineTime.tex — five pgfplots scatter-plot panels,
one per grammar, comparing LL(1)/LR(1) deterministic parsers against
generalised parsers (Earley, GLL, RNGLR, BRNGLR).

Grammars shown are the five with a known LR(1) baseline used in RQ2:
    TinyPascal, S-Expr LL-1, Expr (lr), JSON (lr), TinyC LR-1

Usage:
    python3 bin/make_lr_baseline_plot.py

Output:
    img/lrBaselineTime.tex

Include in the paper as:
    \\begin{figure*}[tp]
      \\centering
      \\input{img/lrBaselineTime.tex}
      \\caption{...}
      \\label{fig:lrBaseline}
    \\end{figure*}

Requires in preamble:
    \\usepackage{pgfplots}
    \\pgfplotsset{compat=1.18}

Design notes
------------
* One minipage per grammar (3 on first row, 2 on second).
* x-axis: token count, log scale, capped at TOKEN_MAX.
* y-axis: time (ms), log scale, per-grammar range (auto ymin/ymax).
* Each parser: scatter markers + power-law regression line.
* LL(1) shown only where available (TinyPascal, S-Expr LL-1).
* CYK and Valiant excluded — they blow up below 200 tokens and would
  compress the interesting region of the plot.
"""

import csv
import math
import os

import numpy as np

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR  = os.path.join(PROJECT_ROOT, "results")
OUTPUT_FILE  = os.path.join(PROJECT_ROOT, "img", "lrBaselineTime.tex")

TOKEN_MAX = 30_000   # clip data to this token count

# Grammar label -> (csv file, has_ll1)
GRAMMARS = [
    ("TinyPascal",   "benchmark_tinypascal.csv", True),
    ("S-Expr LL-1",  "benchmark_sexp_ll1.csv",   True),
    ("Expr (lr)",    "benchmark_calc.csv",        False),
    ("JSON (lr)",    "benchmark_json_lr.csv",     False),
    ("TinyC LR-1",   "benchmark_tinyc_lr.csv",   False),
]

# Parser display name -> (csv name, color, mark, line style)
# Colors match the paper's existing scatter-plot convention.
PARSERS = [
    ("LL(1)",   "LL",    "pink!80!black",      "*",       "densely dashed"),
    ("LR(1)",   "LR",    "gray!70!black",      "square*", "solid"),
    ("Earley",  "Leo",   "green!55!black",     "triangle*","solid"),
    ("GLL",     "GLL",   "red!75!black",       "diamond*","solid"),
    ("RNGLR",   "RNGLR", "violet!70!black",    "*",       "solid"),
    ("BRNGLR",  "BRNGLR","brown!70!black",     "*",       "dotted"),
]

# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def load_csv(filename: str) -> list[dict]:
    path = os.path.join(RESULTS_DIR, filename)
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def get_xy(rows: list[dict], parser_csv: str,
           xmax: int = TOKEN_MAX) -> list[tuple[float, float]]:
    """Return sorted (token_count, time_ms) for successful rows."""
    pts = []
    for r in rows:
        if r["parser"] != parser_csv:
            continue
        if r.get("status", "OK") != "OK":
            continue
        x = int(r["token_count"])
        y = float(r["median_time_ns"]) / 1e6   # ns -> ms
        if x > 0 and y > 0 and x <= xmax:
            pts.append((x, y))
    return sorted(pts)


def fit_loglog(pts: list[tuple]) -> tuple[float, float] | None:
    """Fit log(y) = a*log(x) + b.  Return (a, coeff) or None."""
    if len(pts) < 4:
        return None
    lx = np.log([p[0] for p in pts])
    ly = np.log([p[1] for p in pts])
    a, b = np.polyfit(lx, ly, 1)
    return float(a), float(math.exp(b))


def coords_block(pts: list[tuple]) -> str:
    return " ".join(f"({x:.6g},{y:.6g})" for x, y in pts)


# ---------------------------------------------------------------------------
# Axis-limit helpers (log scale)
# ---------------------------------------------------------------------------

def nice_log_lim(values: list[float], direction: str) -> float:
    """Round a min or max value to a nice power-of-10 boundary."""
    if not values:
        return 1.0 if direction == "min" else 1000.0
    v = min(values) if direction == "min" else max(values)
    exp = math.floor(math.log10(v))
    if direction == "min":
        return 10 ** exp
    else:
        return 10 ** (exp + 1)


# ---------------------------------------------------------------------------
# TikZ / pgfplots generation
# ---------------------------------------------------------------------------

def write_panel(grammar_label: str, rows: list[dict],
                has_ll1: bool, add_legend: bool) -> list[str]:
    """Return lines for one minipage (tikzpicture + axis)."""

    # Collect all y-values to determine axis limits
    all_y: list[float] = []
    all_x: list[float] = []
    for _, csv_name, color, mark, ls in PARSERS:
        if csv_name == "LL" and not has_ll1:
            continue
        pts = get_xy(rows, csv_name)
        all_y += [y for _, y in pts]
        all_x += [x for x, _ in pts]

    xmin_data = min(all_x) if all_x else 1
    xmax_data = min(max(all_x), TOKEN_MAX) if all_x else TOKEN_MAX
    ymin_val  = nice_log_lim(all_y, "min")
    ymax_val  = nice_log_lim(all_y, "max")

    lines: list[str] = [
        r"\begin{tikzpicture}",
        r"\begin{axis}[",
        f"  title={{{grammar_label}}},",
        r"  width=\linewidth,",
        r"  height=5.5cm,",
        r"  xlabel={Tokens},",
        r"  ylabel={Time (ms)},",
        r"  xmode=log,",
        r"  ymode=log,",
        f"  xmin={xmin_data:.4g}, xmax={xmax_data:.4g},",
        f"  ymin={ymin_val:.4g}, ymax={ymax_val:.4g},",
        r"  grid=major,",
        r"  grid style={dashed, gray!30},",
        r"  tick label style={font=\small},",
        r"  label style={font=\small},",
        r"  title style={font=\small\bfseries},",
        r"  clip=true,",
    ]
    if add_legend:
        lines += [
            r"  legend pos=north west,",
            r"  legend style={font=\footnotesize, fill opacity=0.85,"
            r"                 text opacity=1},",
            r"  legend cell align=left,",
        ]
    lines.append(r"]")

    for display_name, csv_name, color, mark, line_style in PARSERS:
        if csv_name == "LL" and not has_ll1:
            continue

        pts = get_xy(rows, csv_name)
        if not pts:
            lines.append(f"% no data for {csv_name}")
            continue

        # Scatter markers
        lines += [
            r"\addplot[",
            f"  color={color}, only marks, mark={mark}, mark size=1.5pt,",
            r"] coordinates {",
            f"  {coords_block(pts)}",
            r"};",
        ]
        if add_legend:
            lines.append(f"\\addlegendentry{{{display_name}}}")

        # Regression line (power-law fit on log-log)
        fit = fit_loglog(pts)
        if fit:
            a, coeff = fit
            xmin_fit = pts[0][0]
            xmax_fit = pts[-1][0]
            lines += [
                r"\addplot[",
                f"  color={color}, thick, {line_style}, no marks, forget plot,",
                f"  domain={xmin_fit:.4g}:{xmax_fit:.4g}, samples=80,",
                f"] {{{coeff:.6g} * x^({a:.5f})}};",
                f"% {grammar_label} {display_name}: n^{a:.2f}",
            ]

    lines += [r"\end{axis}", r"\end{tikzpicture}"]
    return lines


def main() -> None:
    # Load all CSV data
    grammar_rows: dict[str, list[dict]] = {}
    for label, fname, _ in GRAMMARS:
        grammar_rows[label] = load_csv(fname)

    out: list[str] = [
        "% Generated by make_lr_baseline_plot.py — do not edit by hand.",
        "% Caption and label live in the main .tex file.",
    ]

    # Row 1: three grammars
    row1 = GRAMMARS[:3]
    # Row 2: two grammars (centred with \hspace)
    row2 = GRAMMARS[3:]

    minipage_width_3 = "0.32"
    minipage_width_2 = "0.47"

    # --- First row ---
    for i, (label, _, has_ll1) in enumerate(row1):
        add_legend = (i == 0)
        out += [
            f"\\begin{{minipage}}[t]{{{minipage_width_3}\\linewidth}}",
            r"  \centering",
        ]
        out += ["  " + l for l in write_panel(
            label, grammar_rows[label], has_ll1, add_legend)]
        sep = r"\hfill" if i < len(row1) - 1 else ""
        out += [r"\end{minipage}", sep]

    # Spacing between rows
    out.append(r"\par\vspace{4pt}")

    # --- Second row (centred) ---
    out.append(r"\hspace*{0.08\linewidth}")
    for i, (label, _, has_ll1) in enumerate(row2):
        out += [
            f"\\begin{{minipage}}[t]{{{minipage_width_2}\\linewidth}}",
            r"  \centering",
        ]
        out += ["  " + l for l in write_panel(
            label, grammar_rows[label], has_ll1, False)]
        sep = r"\hfill" if i < len(row2) - 1 else ""
        out += [r"\end{minipage}", sep]
    out.append(r"\hspace*{0.08\linewidth}")

    tex = "\n".join(out) + "\n"
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, "w") as fh:
        fh.write(tex)
    print(f"Written: {OUTPUT_FILE}")

    # Summary: fitted exponents
    print("\nFitted log-log exponents (LR-baseline grammars):")
    for label, _, has_ll1 in GRAMMARS:
        rows = grammar_rows[label]
        print(f"  {label}:")
        for display_name, csv_name, *_ in PARSERS:
            if csv_name == "LL" and not has_ll1:
                continue
            pts = get_xy(rows, csv_name)
            fit = fit_loglog(pts)
            if fit:
                a, coeff = fit
                print(f"    {display_name:8s}  n^{a:.2f}  "
                      f"(n={pts[0][0]}..{pts[-1][0]}, {len(pts)} pts)")
            else:
                print(f"    {display_name:8s}  insufficient data ({len(pts)} pts)")


if __name__ == "__main__":
    main()
