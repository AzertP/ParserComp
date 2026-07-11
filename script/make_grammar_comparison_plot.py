#!/usr/bin/env python3
"""
make_grammar_comparison_plot.py
--------------------------------
Generate img/grammarComparisonTime.tex — two pgfplots panels side by side,
one for JSON and one for Expr, each comparing three grammar/parser pairings:

    LL(1) grammar  + LL(1) parser
    LR(1) grammar  + LR(1) parser
    Ambiguous grammar + BRNGLR parser

File/parser mappings
--------------------
JSON:
    LL(1) grammar  : benchmark_json_ll1.csv   parser=LL
    LR(1) grammar  : benchmark_json.csv        parser=LR   (right-recursive, LR(1))
    Ambi + BRNGLR  : benchmark_json_ambi.csv   parser=BRNGLR

Expr:
    LL(1) grammar  : benchmark_calc_ll1.csv    parser=LL
    LR(1) grammar  : benchmark_calc.csv        parser=LR   (left-recursive, LR(1))
    Ambi + BRNGLR  : benchmark_expr_ambi.csv   parser=BRNGLR

Note: Expr ambi data is only available up to ~500 tokens; both Expr plots
therefore clip to TOKEN_MAX_EXPR. JSON data extends to 30k tokens.

Usage:
    python3 bin/make_grammar_comparison_plot.py

Output:
    img/grammarComparisonTime.tex

Include in the paper as:
    \\begin{figure*}[tp]
      \\centering
      \\input{img/grammarComparisonTime.tex}
      \\caption{...}
      \\label{fig:grammarComparison}
    \\end{figure*}

Requires in preamble:
    \\usepackage{pgfplots}
    \\pgfplotsset{compat=1.18}
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
OUTPUT_FILE  = os.path.join(PROJECT_ROOT, "img", "grammarComparisonTime.tex")

TOKEN_MAX = 30_000   # clip all data to this token count

# Shared axis parameters — identical to the edited lrBaselineTime.tex
YMIN = 0.001
YMAX = 1000

# Three series — (display label, csv file, parser csv name, color, mark, linestyle)
SERIES_JSON = [
    ("LL(1) + LL(1)",  "benchmark_json_ll1.csv",  "LL",     "pink!80!black",  "*",        "densely dashed"),
    ("LR(1) + LR(1)",  "benchmark_json.csv",       "LR",     "gray!70!black",  "square",   "solid"),
    ("Ambi + BRNGLR",  "benchmark_json_ambi.csv",  "BRNGLR", "brown!70!black", "triangle*","dotted"),
]

SERIES_EXPR = [
    ("LL(1) + LL(1)",  "benchmark_calc_ll1.csv",  "LL",     "pink!80!black",  "*",        "densely dashed"),
    ("LR(1) + LR(1)",  "benchmark_calc.csv",       "LR",     "gray!70!black",  "square",   "solid"),
    ("Ambi + BRNGLR",  "benchmark_expr_ambi.csv",  "BRNGLR", "brown!70!black", "triangle*","dotted"),
]

# ---------------------------------------------------------------------------
# Data helpers (identical to make_cyk_valiant_plot.py style)
# ---------------------------------------------------------------------------

def load_csv(filename: str) -> list[dict]:
    path = os.path.join(RESULTS_DIR, filename)
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def get_xy(rows: list[dict], parser_csv: str,
           xmax: int = TOKEN_MAX) -> list[tuple[float, float]]:
    """Return sorted (token_count, time_ms) for successful rows within xmax."""
    pts = []
    for r in rows:
        if r["parser"] != parser_csv:
            continue
        if r.get("status", "OK") != "OK":
            continue
        x = int(r["token_count"])
        y = float(r["median_time_ns"]) / 1e6
        if x > 0 and y > 0 and x <= xmax:
            pts.append((x, y))
    return sorted(pts)


def fit_loglog(pts: list[tuple]) -> tuple[float, float] | None:
    """Fit log(y) = a*log(x) + b on positive-y points. Return (a, coeff) or None."""
    pos = [(x, y) for x, y in pts if y > 0]
    if len(pos) < 4:
        return None
    lx = np.log([p[0] for p in pos])
    ly = np.log([p[1] for p in pos])
    a, b = np.polyfit(lx, ly, 1)
    return float(a), float(math.exp(b))


def coords_block(pts: list[tuple]) -> str:
    return " ".join(f"({x:.6g},{y:.6g})" for x, y in pts)


def fit_domain_xmax(coeff: float, a: float, ymax: float,
                    data_xmax: float) -> float:
    """Clip regression line domain so it stays below ymax."""
    if coeff <= 0 or a <= 0:
        return data_xmax
    try:
        x_at_ymax = (ymax / coeff) ** (1.0 / a)
    except (ZeroDivisionError, ValueError):
        return data_xmax
    return min(data_xmax, x_at_ymax)

# ---------------------------------------------------------------------------
# Axis generation (one panel)
# ---------------------------------------------------------------------------

def write_panel(
    title: str,
    series: list,
    add_legend: bool,
) -> list[str]:
    """Return lines for one tikzpicture+axis block (no figure wrapper)."""

    # Compute data-driven x limits across all series
    all_x: list[float] = []
    for _, fname, parser_csv, *_ in series:
        rows = load_csv(fname)
        pts  = get_xy(rows, parser_csv)
        all_x += [x for x, _ in pts]

    xmin_data = min(all_x) if all_x else 1
    xmax_data = max(all_x) if all_x else TOKEN_MAX

    lines: list[str] = [
        r"\begin{tikzpicture}",
        r"\begin{axis}[",
        f"  title={{{title}}},",
        r"  width=\linewidth,",
        r"  height=5.5cm,",
        r"  xlabel={Input size (tokens)},",
        r"  ylabel={Time (ms)},",
        r"  xmode=log,",
        r"  ymode=log,",
        f"  xmin={xmin_data:.4g}, xmax={xmax_data:.4g},",
        f"  ymin={YMIN}, ymax={YMAX},",
        r"  grid=major,",
        r"  grid style={dashed, gray!30},",
        r"  tick label style={font=\small},",
        r"  label style={font=\small},",
        r"  title style={font=\small\bfseries, at={(0.5,1)}, anchor=north,"
        r"                 yshift=-2mm, fill=white, fill opacity=0.75,"
        r"                 text opacity=1, inner sep=2pt},",
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

    for display_name, fname, parser_csv, color, mark, line_style in series:
        rows = load_csv(fname)
        pts  = get_xy(rows, parser_csv)

        if not pts:
            lines.append(f"% no data for {display_name}")
            continue

        xmin_g = pts[0][0]
        xmax_g = pts[-1][0]

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

        # Power-law regression line
        fit = fit_loglog(pts)
        if fit:
            a, coeff = fit
            domain_end   = fit_domain_xmax(coeff, a, YMAX, xmax_g)
            domain_start = max(1, xmin_g)
            if domain_end > domain_start:
                lines += [
                    r"\addplot[",
                    f"  color={color}, thick, {line_style}, no marks, forget plot,",
                    f"  domain={domain_start:.4g}:{domain_end:.4g}, samples=80,",
                    f"] {{{coeff:.6g} * x^({a:.5f})}};",
                    f"% {title} {display_name}: n^{a:.2f}",
                ]

    lines += [r"\end{axis}", r"\end{tikzpicture}"]
    return lines


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    out: list[str] = [
        "% Generated by make_grammar_comparison_plot.py — do not edit by hand.",
        "% Caption and label live in the main .tex file.",
    ]

    panels = [
        ("JSON", SERIES_JSON, True),
        ("Expr", SERIES_EXPR, False),
    ]

    # Side-by-side minipages (matches smallCykValiantTime.tex layout)
    out += [
        r"\begin{minipage}[t]{0.47\linewidth}",
        r"  \centering",
    ]
    out += ["  " + l for l in write_panel(*panels[0])]
    out += [
        r"\end{minipage}",
        r"\hfill",
        r"\begin{minipage}[t]{0.47\linewidth}",
        r"  \centering",
    ]
    out += ["  " + l for l in write_panel(*panels[1])]
    out += [r"\end{minipage}"]

    tex = "\n".join(out) + "\n"
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, "w") as fh:
        fh.write(tex)
    print(f"Written: {OUTPUT_FILE}")

    # Summary
    print("\nFitted power-law exponents:")
    for title, series, _ in panels:
        print(f"  {title}:")
        for display_name, fname, parser_csv, *_ in series:
            rows = load_csv(fname)
            pts  = get_xy(rows, parser_csv)
            fit  = fit_loglog(pts)
            if fit:
                a, coeff = fit
                print(f"    {display_name:20s}  n^{a:.2f}  "
                      f"(n={pts[0][0]}..{pts[-1][0]}, {len(pts)} pts)")
            else:
                print(f"    {display_name:20s}  insufficient data ({len(pts)} pts)")


if __name__ == "__main__":
    main()
