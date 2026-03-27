#!/usr/bin/env python3
"""
make_cyk_valiant_plot.py
------------------------
Generate img/smallCykValiantTime.tex — a pgfplots side-by-side figure
comparing CYK (left) and Valiant (right) runtime on small inputs across
all six grammars for which both parsers were benchmarked.

Usage:
    python3 make_cyk_valiant_plot.py

Output:
    img/smallCykValiantTime.tex

Include in the paper as:
    \\begin{figure*}[tp]
      \\centering
      \\input{img/smallCykValiantTime.tex}
      \\caption{...}
      \\label{fig:cykValiant}
    \\end{figure*}

Requires in preamble:
    \\usepackage{pgfplots}
    \\pgfplotsset{compat=1.18}

Axis convention (both plots share identical axes):
    x : 0–140  tokens,  ticks every 20
    y : 0–300  ms,      ticks every 50
Points beyond these limits are clipped by pgfplots.
"""

import csv
import math
import os

import numpy as np

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))  # bin/../
RESULTS_DIR  = os.path.join(PROJECT_ROOT, "results")
OUTPUT_FILE  = os.path.join(PROJECT_ROOT, "img", "smallCykValiantTime.tex")

# All grammars with CYK and Valiant data
GRAMMARS = [
    ("S-Expr",     "benchmark_sexp.csv"),
    ("Calculator", "benchmark_calc.csv"),
    ("TinyC",      "benchmark_tinyc.csv"),
    ("JSON",       "benchmark_json.csv"),
    ("Bool",       "benchmark_bool.csv"),
    ("Expr-Ambi",  "benchmark_expr_ambi.csv"),
]

# One colour per grammar — all use the same circle marker
COLORS = [
    "blue!80!black",
    "red!75!black",
    "green!55!black",
    "orange!80!black",
    "violet!70!black",
    "cyan!60!black",
]

# Shared axis limits and tick positions
XMIN, XMAX   = 0, 140
YMIN, YMAX   = 0, 300
XTICK = "{0,20,...,140}"
YTICK = "{0,50,...,300}"

# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def load_csv(filename: str) -> list[dict]:
    path = os.path.join(RESULTS_DIR, filename)
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def get_xy(rows: list[dict], parser: str) -> list[tuple[float, float]]:
    """Return sorted (x_tokens, y_ms) pairs for successful rows of a parser."""
    pts = []
    for r in rows:
        if r["parser"] != parser:
            continue
        if r.get("success", "true") != "true":
            continue
        x = int(r["input_length"])
        y = float(r["median_time_ns"]) / 1e6
        if x > 0 and y > 0:
            pts.append((x, y))
    return sorted(pts)


def fit_loglog(pts: list[tuple]) -> tuple[float, float] | None:
    """
    Fit log(y) = a*log(x) + b.  Returns (a, b) or None if < 3 points.
    """
    if len(pts) < 3:
        return None
    lx = np.log([x for x, _ in pts])
    ly = np.log([y for _, y in pts])
    a, b = np.polyfit(lx, ly, 1)
    return float(a), float(b)


# ---------------------------------------------------------------------------
# TikZ / pgfplots generation
# ---------------------------------------------------------------------------

def coords_block(pts: list[tuple]) -> str:
    return " ".join(f"({x},{y:.6g})" for x, y in pts)


def fit_domain_xmax(coeff: float, a: float, ymax: float,
                    data_xmax: float) -> float:
    """
    Upper x limit for the regression line so that y stays below ymax,
    avoiding FPU overflow in pgfplots for steep (Valiant) curves.
    """
    try:
        x_at_ymax = (ymax / coeff) ** (1.0 / a)
    except (ZeroDivisionError, ValueError):
        x_at_ymax = data_xmax
    return min(data_xmax, x_at_ymax)


def write_axis(
    parser: str,
    title: str,
    grammar_data: dict,
    add_legend: bool,
) -> list[str]:
    """Return lines for one tikzpicture + axis block."""

    lines: list[str] = [
        r"\begin{tikzpicture}",
        r"\begin{axis}[",
        f"  title={{{title}}},",
        r"  width=\linewidth,",
        r"  height=6cm,",
        r"  xlabel={Input size (tokens)},",
        r"  ylabel={Time (ms)},",
        f"  xmin={XMIN}, xmax={XMAX},",
        f"  ymin={YMIN}, ymax={YMAX},",
        f"  xtick={XTICK},",
        f"  ytick={YTICK},",
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
            r"  legend style={font=\footnotesize},",
            r"  legend cell align=left,",
        ]
    lines.append(r"]")

    for (label, _), color in zip(GRAMMARS, COLORS):
        pts = get_xy(grammar_data[label], parser)
        # Keep only points within axis range for a clean plot
        pts_in = [(x, y) for x, y in pts if x <= XMAX]
        if not pts_in:
            lines.append(f"% no {parser} data for {label} within x<={XMAX}")
            continue

        xmin_g = pts_in[0][0]
        xmax_g = pts_in[-1][0]

        # Scatter markers only — no connecting line, same circle for all grammars
        lines += [
            r"\addplot[",
            f"  color={color}, only marks, mark=*, mark size=1.5pt,",
            r"] coordinates {",
            f"  {coords_block(pts_in)}",
            r"};",
        ]
        if add_legend:
            lines.append(f"\\addlegendentry{{{label}}}")

        # Smooth regression curve — thick solid line, no marks
        fit = fit_loglog(pts)
        if fit:
            a, b = fit
            coeff = math.exp(b)
            domain_end = fit_domain_xmax(coeff, a, YMAX, xmax_g)
            domain_start = max(1, xmin_g)
            if domain_end > domain_start:
                lines += [
                    r"\addplot[",
                    f"  color={color}, thick, solid, no marks, forget plot,",
                    f"  domain={domain_start:.4g}:{domain_end:.4g}, samples=60,",
                    f"] {{{coeff:.6g} * x^({a:.5f})}};",
                    f"% {label} {parser}: n^{a:.2f}",
                ]

    lines += [r"\end{axis}", r"\end{tikzpicture}"]
    return lines


def main() -> None:
    grammar_data: dict[str, list[dict]] = {}
    for label, fname in GRAMMARS:
        grammar_data[label] = load_csv(fname)

    out: list[str] = [
        "% Generated by make_cyk_valiant_plot.py — do not edit by hand.",
        r"\begin{minipage}[t]{0.47\linewidth}",
        r"  \centering",
    ]
    out += ["  " + l for l in write_axis("CYK", "CYK", grammar_data, add_legend=True)]
    out += [
        r"\end{minipage}",
        r"\hfill",
        r"\begin{minipage}[t]{0.47\linewidth}",
        r"  \centering",
    ]
    out += ["  " + l for l in write_axis("Valiant", "Valiant", grammar_data, add_legend=False)]
    out += [r"\end{minipage}"]

    tex = "\n".join(out) + "\n"
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, "w") as fh:
        fh.write(tex)
    print(f"Written: {OUTPUT_FILE}")

    # Summary
    print("\nFitted log-log exponents (all data, not clipped):")
    for parser in ("CYK", "Valiant"):
        print(f"  {parser}:")
        for label, fname in GRAMMARS:
            pts = get_xy(grammar_data[label], parser)
            fit = fit_loglog(pts)
            pts_in = [(x, y) for x, y in pts if x <= XMAX]
            if fit:
                a, b = fit
                coeff = math.exp(b)
                clip_note = (f", {len(pts)-len(pts_in)} pts clipped"
                             if len(pts) > len(pts_in) else "")
                print(f"    {label:12s}  n^{a:.2f}  "
                      f"(n={pts[0][0]}..{pts[-1][0]}{clip_note})")
            else:
                print(f"    {label:12s}  insufficient data")


if __name__ == "__main__":
    main()
