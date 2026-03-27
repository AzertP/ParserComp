#!/usr/bin/env python3
"""
make_all_grammars_general_plot.py
----------------------------------
Generate img/allGrammarsGeneralTime.tex — scatter-plot panels for all 22
grammars, showing only the four practical generalized parsers:
Earley, GLL, RNGLR, BRNGLR.

Layout: 4 panels per row, 4 per row × 6 rows (last row has 2 panels).

Usage:
    python3 bin/make_all_grammars_general_plot.py

Output:
    img/allGrammarsGeneralTime.tex
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
OUTPUT_FILE  = os.path.join(PROJECT_ROOT, "img", "allGrammarsGeneralTime.tex")

TOKEN_MAX = 30_000

# All 22 grammars: (display label, csv filename)
GRAMMARS = [
    ("TinyPascal",    "benchmark_tinypascal.csv"),
    ("S-Expr LL-1",   "benchmark_sexp_ll1.csv"),
    ("Expr LL-1",     "benchmark_calc_ll1.csv"),
    ("JSON LL-1",     "benchmark_json_ll1.csv"),
    ("Expr (lr)",     "benchmark_calc.csv"),
    ("Expr (rr)",     "benchmark_expr.csv"),
    ("JSON (rr)",     "benchmark_json.csv"),
    ("JSON (lr)",     "benchmark_json_lr.csv"),
    ("TinyC LR-1",    "benchmark_tinyc_lr.csv"),
    ("S-Expression",  "benchmark_sexp.csv"),
    ("Bool",          "benchmark_bool.csv"),
    ("Expr (ambig)",  "benchmark_expr_ambi.csv"),
    ("JSON (ambig)",  "benchmark_json_ambi.csv"),
    ("TinyC",         "benchmark_tinyc.csv"),
    ("ANSI C",        "benchmark_ansi_c.csv"),
    ("Pascal",        "benchmark_pascal.csv"),
    ("Java",          "benchmark_java.csv"),
    ("C++",           "benchmark_cpp.csv"),
    ("CSS",           "benchmark_css.csv"),
    ("HTML",          "benchmark_html.csv"),
    ("Shell",         "benchmark_shell.csv"),
    ("SQL",           "benchmark_sql.csv"),
]

# Only the four practical generalized parsers.
# csv_name is the value in the "parser" column of the CSV.
PARSERS = [
    ("Earley",  "Leo",    "green!55!black",  "triangle*", "solid"),
    ("GLL",     "GLL",    "red!75!black",    "diamond*",  "solid"),
    ("RNGLR",   "RNGLR",  "violet!70!black", "*",         "solid"),
    ("BRNGLR",  "BRNGLR", "brown!70!black",  "*",         "dotted"),
]

PANELS_PER_ROW = 4
PANEL_WIDTH    = "0.235"   # linewidth fraction; 4 × 0.235 + 3 × hfill ≈ full width
PANEL_HEIGHT   = "4.5cm"

# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def load_csv(filename: str) -> list[dict]:
    path = os.path.join(RESULTS_DIR, filename)
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def get_xy(rows: list[dict], parser_csv: str,
           xmax: int = TOKEN_MAX) -> list[tuple[float, float]]:
    pts = []
    for r in rows:
        if r["parser"] != parser_csv:
            continue
        if r.get("status", "OK") != "OK":
            continue
        x = int(r["token_count"])
        y = float(r["median_time_ns"]) / 1e6   # ns → ms
        if x > 0 and y > 0 and x <= xmax:
            pts.append((x, y))
    return sorted(pts)


def fit_loglog(pts: list[tuple]) -> tuple[float, float] | None:
    if len(pts) < 4:
        return None
    lx = np.log([p[0] for p in pts])
    ly = np.log([p[1] for p in pts])
    a, b = np.polyfit(lx, ly, 1)
    return float(a), float(math.exp(b))


def coords_block(pts: list[tuple]) -> str:
    return " ".join(f"({x:.6g},{y:.6g})" for x, y in pts)


def nice_log_lim(values: list[float], direction: str) -> float:
    if not values:
        return 1.0 if direction == "min" else 1000.0
    v = min(values) if direction == "min" else max(values)
    exp = math.floor(math.log10(v))
    return 10 ** exp if direction == "min" else 10 ** (exp + 1)


# ---------------------------------------------------------------------------
# TikZ / pgfplots generation
# ---------------------------------------------------------------------------

def write_panel(grammar_label: str, rows: list[dict],
                add_legend: bool) -> list[str]:
    all_y: list[float] = []
    all_x: list[float] = []
    for _, csv_name, *_ in PARSERS:
        pts = get_xy(rows, csv_name)
        all_y += [y for _, y in pts]
        all_x += [x for x, _ in pts]

    if not all_x:
        return [f"% no data for {grammar_label}"]

    xmin_data = min(all_x)
    xmax_data = min(max(all_x), TOKEN_MAX)
    ymin_val  = nice_log_lim(all_y, "min")
    ymax_val  = nice_log_lim(all_y, "max")

    lines: list[str] = [
        r"\begin{tikzpicture}",
        r"\begin{axis}[",
        f"  title={{{grammar_label}}},",
        r"  width=\linewidth,",
        f"  height={PANEL_HEIGHT},",
        r"  xlabel={Tokens},",
        r"  ylabel={Time (ms)},",
        r"  xmode=log,",
        r"  ymode=log,",
        f"  xmin={xmin_data:.4g}, xmax={xmax_data:.4g},",
        f"  ymin={ymin_val:.4g}, ymax={ymax_val:.4g},",
        r"  grid=major,",
        r"  grid style={dashed, gray!30},",
        r"  tick label style={font=\tiny},",
        r"  label style={font=\tiny},",
        r"  title style={font=\small\bfseries},",
        r"  clip=true,",
    ]
    if add_legend:
        lines += [
            r"  legend pos=north west,",
            r"  legend style={font=\tiny, fill opacity=0.85, text opacity=1},",
            r"  legend cell align=left,",
        ]
    lines.append(r"]")

    for display_name, csv_name, color, mark, line_style in PARSERS:
        pts = get_xy(rows, csv_name)
        if not pts:
            lines.append(f"% no data for {csv_name}")
            continue

        lines += [
            r"\addplot[",
            f"  color={color}, only marks, mark={mark}, mark size=1pt,",
            r"] coordinates {",
            f"  {coords_block(pts)}",
            r"};",
        ]
        if add_legend:
            lines.append(f"\\addlegendentry{{{display_name}}}")

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
    # Pre-load all CSV data
    grammar_rows: dict[str, list[dict]] = {}
    for label, fname in GRAMMARS:
        grammar_rows[label] = load_csv(fname)

    out: list[str] = [
        "% Generated by make_all_grammars_general_plot.py — do not edit by hand.",
        "% Caption and label live in the main .tex file.",
    ]

    # Chunk grammars into rows of PANELS_PER_ROW
    chunks = [GRAMMARS[i:i+PANELS_PER_ROW]
              for i in range(0, len(GRAMMARS), PANELS_PER_ROW)]

    first_panel = True
    for chunk_idx, chunk in enumerate(chunks):
        # Centre the last (partial) row
        n = len(chunk)
        if n < PANELS_PER_ROW:
            padding = (PANELS_PER_ROW - n) / 2
            out.append(f"\\hspace*{{{padding:.2f}\\linewidth}}")

        for i, (label, _) in enumerate(chunk):
            add_legend = first_panel
            first_panel = False

            out += [
                f"\\begin{{minipage}}[t]{{{PANEL_WIDTH}\\linewidth}}",
                r"  \centering",
            ]
            out += ["  " + l for l in write_panel(
                label, grammar_rows[label], add_legend)]
            sep = r"\hfill" if i < n - 1 else ""
            out += [r"\end{minipage}", sep]

        if n < PANELS_PER_ROW:
            out.append(f"\\hspace*{{{padding:.2f}\\linewidth}}")

        if chunk_idx < len(chunks) - 1:
            out.append(r"\par\vspace{4pt}")

    preamble = [
        r"% Standalone document — compile with: latexmk -pdf",
        r"% Generated by make_all_grammars_general_plot.py — do not edit by hand.",
        r"\documentclass[a4paper]{article}",
        r"\usepackage{pgfplots}",
        r"\pgfplotsset{compat=1.18}",
        r"\usepackage{tikz}",
        r"\usepackage[margin=1.5cm, landscape]{geometry}",
        r"% Increase TeX memory pools for large coordinate data",
        r"\usepackage{pgfplotstable}",
        r"\begin{document}",
        r"\pagestyle{empty}",
        r"\noindent\textbf{Runtime (ms) vs.\ token count for all 22 benchmark grammars}\\",
        r"Earley {\small(green $\triangle$)}, GLL {\small(red $\diamond$)},",
        r"RNGLR {\small(violet $\bullet$)}, BRNGLR {\small(brown $\cdot$)}.",
        r"Both axes logarithmic; lines are power-law fits. CYK and Valiant excluded.",
        r"\medskip",
        r"",
    ]

    postamble = [
        r"",
        r"\end{document}",
    ]

    tex = "\n".join(preamble + out + postamble) + "\n"
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, "w") as fh:
        fh.write(tex)
    print(f"Written: {OUTPUT_FILE}")

    # Print fitted exponents summary
    print("\nFitted log-log exponents (all grammars, generalized parsers only):")
    for label, _ in GRAMMARS:
        rows = grammar_rows[label]
        print(f"  {label}:")
        for display_name, csv_name, *_ in PARSERS:
            pts = get_xy(rows, csv_name)
            fit = fit_loglog(pts)
            if fit:
                a, coeff = fit
                print(f"    {display_name:8s}  n^{a:.2f}  ({len(pts)} pts)")
            else:
                print(f"    {display_name:8s}  insufficient data ({len(pts)} pts)")


if __name__ == "__main__":
    main()
