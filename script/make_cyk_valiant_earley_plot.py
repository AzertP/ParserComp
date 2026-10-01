#!/usr/bin/env python3
"""
make_cyk_valiant_earley_plot.py
---------------------------
Generate img/cykValiantEarleyTime.tex — log--log runtime panels comparing
the two offline generalized parsers, CYK and Valiant, against Earley on
the five grammars for which all three were benchmarked.

Unlike the summary table, which aggregates into input-size buckets, this
plots every successful measurement, so the crossover between CYK and
Valiant and the point at which each parser stops finishing within the
time budget are both visible.

Usage:
    python3 script/make_cyk_valiant_earley_plot.py

Output:
    img/cykValiantEarleyTime.tex

Include in the paper as:
    \\begin{figure*}[tp]
      \\centering
      \\input{img/cykValiantEarleyTime.tex}
      \\caption{...}
      \\label{fig:cykValiantEarleyTime}
    \\end{figure*}

Requires in preamble:
    \\usepackage{pgfplots}
    \\pgfplotsset{compat=1.18}
"""

import csv
import math
import os

from plot_utils import fit_power_law

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

from artifact_paths import RESULT_ROOT, IMG_DIR
RESULTS_DIR = os.path.join(RESULT_ROOT, "benchmark_csv")
OUTPUT_FILE = os.path.join(IMG_DIR, "cykValiantEarleyTime.tex")
# Single representative panel, set beside the summary table in the main text.
OUTPUT_ONE = os.path.join(IMG_DIR, "cykValiantEarleyOne.tex")

# The grammar whose panel stands in for the set. Expr (lr) sits at the
# cross-grammar median on every ratio the table reports and on all three
# fitted exponents, and it is the grammar the RQ1 prose quotes, so the
# numbers in the text can be read straight off this panel.
REPRESENTATIVE = "Expr (lr)"

# The five grammars carried by the CYK/Valiant table, in the same order.
GRAMMARS = [
    ("S-Expr LR-1", "benchmark_sexp.csv"),
    ("Expr (lr)", "benchmark_calc.csv"),
    ("Bool", "benchmark_bool.csv"),
    ("Expr (ambig)", "benchmark_expr_ambi.csv"),
    ("Expr (rr)", "benchmark_expr.csv"),
]

# (display name, parser name in CSV, color, mark)
PARSERS = [
    ("CYK", "CYK", "blue!80!black", "square*"),
    ("Valiant", "Valiant", "orange!80!black", "triangle*"),
    ("Earley", "Leo", "green!55!black", "*"),
]

# Earley keeps going long after both offline parsers have timed out.
# Plotting its full range would leave the region of interest squeezed into
# the left third of every panel, so the x-axis stops an octave past the
# largest input any offline parser completed.
XMAX_HEADROOM = 2.0

MINIPAGE_WIDTH_3 = "0.32"
MINIPAGE_WIDTH_2 = "0.47"
# Side padding that centers the two-panel second row: 2*0.47 + 2*PAD <= 1.
ROW2_PAD = "0.02"

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_csv(filename):
    path = os.path.join(RESULTS_DIR, filename)
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def get_xy(rows, parser_csv, xmax=None):
    """Return sorted (token_count, time_ms) for successful rows."""
    pts = []
    for r in rows:
        if r["parser"] != parser_csv:
            continue
        if (
            r.get("status", "OK") != "OK"
            or r.get("recognized", "true") != "true"
            or r.get("parse_correct", "true") != "true"
        ):
            continue
        x = int(r["token_count"])
        y = float(r["median_time_ns"]) / 1e6   # ns -> ms
        if x > 0 and y > 0 and (xmax is None or x <= xmax):
            pts.append((x, y))
    return sorted(pts)


def offline_xmax(rows):
    """Largest input either offline parser completed, plus headroom."""
    reach = []
    for _, csv_name, _, _ in PARSERS:
        if csv_name == "Leo":
            continue
        pts = get_xy(rows, csv_name)
        if pts:
            reach.append(pts[-1][0])
    return max(reach) * XMAX_HEADROOM if reach else None


def coords_block(pts):
    return " ".join(f"({x:.6g},{y:.6g})" for x, y in pts)


def nice_log_lim(values, direction):
    """Round a min or max value to a nice power-of-10 boundary."""
    if not values:
        return 1.0 if direction == "min" else 1000.0
    v = min(values) if direction == "min" else max(values)
    exp = math.floor(math.log10(v))
    return 10 ** exp if direction == "min" else 10 ** (exp + 1)


# ---------------------------------------------------------------------------
# TikZ / pgfplots generation
# ---------------------------------------------------------------------------


def write_panel(grammar_label, rows, add_legend, show_xlabel=True,
                compact=False):
    """Return lines for one panel (tikzpicture + axis).

    With compact=True the panel is sized for the narrow column beside
    \\Cref{tab:cykValiant}: no title, since the caption names the grammar,
    and smaller marks so the three series stay separable.
    """
    xmax = offline_xmax(rows)
    series = [(d, c, col, mk, get_xy(rows, c, xmax))
              for d, c, col, mk in PARSERS]

    all_x = [x for *_, pts in series for x, _ in pts]
    all_y = [y for *_, pts in series for _, y in pts]
    xmin_data = min(all_x) if all_x else 1
    xmax_data = max(all_x) if all_x else 1000

    lines = [
        r"\begin{tikzpicture}",
        r"\begin{axis}[",
        (r"  title={}," if compact else f"  title={{{grammar_label}}},"),
        r"  width=\linewidth,",
        (r"  height=4.7cm," if compact else r"  height=4.6cm,"),
        (r"  xlabel={Tokens}," if show_xlabel else r"  xlabel={},"),
        r"  ylabel={Time (ms)},",
        r"  xmode=log,",
        r"  ymode=log,",
        f"  xmin={xmin_data:.4g}, xmax={xmax_data:.4g},",
        f"  ymin={nice_log_lim(all_y, 'min'):.4g},"
        f" ymax={nice_log_lim(all_y, 'max'):.4g},",
        r"  grid=major,",
        r"  grid style={dashed, gray!30},",
        r"  tick label style={font=\tiny},",
        r"  label style={font=\tiny},",
        r"  title style={font=\small\bfseries, at={(0.5,1)}, anchor=north,"
        r"                 yshift=-2mm, fill=white, fill opacity=0.75,"
        r"                 text opacity=1, inner sep=2pt},",
        r"  clip=true,",
    ]
    if add_legend:
        lines += [
            r"  legend pos=north west,",
            r"  legend style={font=\tiny, fill opacity=0.85,"
            r"                 text opacity=1},",
            r"  legend cell align=left,",
        ]
    lines.append(r"]")

    for display_name, csv_name, color, mark, pts in series:
        if not pts:
            lines.append(f"% no data for {csv_name}")
            continue

        lines += [
            r"\addplot[",
            f"  color={color}, only marks, mark={mark},"
            f" mark size={0.7 if compact else 1}pt,",
            r"] coordinates {",
            f"  {coords_block(pts)}",
            r"};",
        ]
        if add_legend:
            lines.append(f"\\addlegendentry{{{display_name}}}")

    lines += [r"\end{axis}", r"\end{tikzpicture}"]
    return lines


def main():
    grammar_rows = {label: load_csv(fname) for label, fname in GRAMMARS}

    out = [
        "% Generated by script/make_cyk_valiant_earley_plot.py — do not edit by hand.",
        "% Caption and label live in the main .tex file.",
    ]

    # Three panels on the first row, the remaining two centered below.
    row1 = GRAMMARS[:3]
    row2 = GRAMMARS[3:]

    for i, (label, _) in enumerate(row1):
        out += [
            f"\\begin{{minipage}}[t]{{{MINIPAGE_WIDTH_3}\\linewidth}}",
            r"  \centering",
        ]
        out += ["  " + l for l in
                write_panel(label, grammar_rows[label], add_legend=(i == 0))]
        out += [r"\end{minipage}", r"\hfill" if i < len(row1) - 1 else ""]

    out += [r"\par\vspace{4pt}", f"\\hspace*{{{ROW2_PAD}\\linewidth}}"]

    for i, (label, _) in enumerate(row2):
        out += [
            f"\\begin{{minipage}}[t]{{{MINIPAGE_WIDTH_2}\\linewidth}}",
            r"  \centering",
        ]
        out += ["  " + l for l in
                write_panel(label, grammar_rows[label], add_legend=False)]
        out += [r"\end{minipage}", r"\hfill" if i < len(row2) - 1 else ""]

    out.append(f"\\hspace*{{{ROW2_PAD}\\linewidth}}")

    tex = "\n".join(out) + "\n"
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, "w") as fh:
        fh.write(tex)
    print(f"Written: {OUTPUT_FILE}")

    # Single representative panel for the main text. Bare tikzpicture, so
    # the enclosing float in the .tex file controls its width.
    one = [
        "% Generated by script/make_cyk_valiant_earley_plot.py — do not edit by hand.",
        "% Caption and label live in the main .tex file.",
        f"% Representative grammar: {REPRESENTATIVE}.",
    ]
    one += write_panel(REPRESENTATIVE, grammar_rows[REPRESENTATIVE],
                       add_legend=True, compact=True)
    with open(OUTPUT_ONE, "w") as fh:
        fh.write("\n".join(one) + "\n")
    print(f"Written: {OUTPUT_ONE}")

    # Reported for the prose only; the panels show the raw scatter.
    print("\nFitted log-log exponents and reach:")
    for label, _ in GRAMMARS:
        rows = grammar_rows[label]
        xmax = offline_xmax(rows)
        print(f"  {label}:")
        for display_name, csv_name, _, _ in PARSERS:
            pts = get_xy(rows, csv_name, xmax)
            fit = fit_power_law(pts, min_points=4)
            if not fit:
                continue
            full = get_xy(rows, csv_name)
            print(f"    {display_name:8s} n^{fit[0]:.2f}  "
                  f"({len(pts)} pts plotted, max {full[-1][0]} tokens)")


if __name__ == "__main__":
    main()
