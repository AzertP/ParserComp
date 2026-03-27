#!/usr/bin/env python3
"""Generate img/grammarComparisonBRNGLR.tex

Two side-by-side panels (JSON, Expr) each showing BRNGLR running on three
grammar forms that accept the same language:
  - LL(1)-refactored grammar
  - LR(1) grammar
  - Ambiguous grammar

This isolates the effect of grammar form on BRNGLR performance.
"""

import csv, math, os
import numpy as np

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
OUTPUT_FILE = os.path.join(os.path.dirname(__file__), "..", "img",
                           "grammarComparisonBRNGLR.tex")

TOKEN_MAX = 60_000   # clip ceiling (all data fits within this)

YMIN = 0.001
YMAX = 1000

# Three series per panel: (display label, csv file, color, mark, linestyle)
# All use parser=BRNGLR from their respective CSV.
SERIES_JSON = [
    ("LL(1) grammar", "benchmark_json_ll1.csv",  "teal!70!black",   "*",        "densely dashed"),
    ("LR(1) grammar", "benchmark_json.csv",       "gray!70!black",   "square",   "solid"),
    ("Ambi grammar",  "benchmark_json_ambi.csv",  "orange!80!black", "triangle*","dotted"),
]

SERIES_EXPR = [
    ("LL(1) grammar", "benchmark_calc_ll1.csv",  "teal!70!black",   "*",        "densely dashed"),
    ("LR(1) grammar", "benchmark_calc.csv",       "gray!70!black",   "square",   "solid"),
    ("Ambi grammar",  "benchmark_expr_ambi.csv",  "orange!80!black", "triangle*","dotted"),
]

# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def load_csv(filename: str) -> list[dict]:
    path = os.path.join(RESULTS_DIR, filename)
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def get_xy(rows: list[dict], xmax: int = TOKEN_MAX) -> list[tuple[float, float]]:
    """Return sorted (token_count, time_ms) for BRNGLR OK rows within xmax."""
    pts = []
    for r in rows:
        if r["parser"] != "BRNGLR":
            continue
        if r.get("status", "OK") != "OK":
            continue
        x = int(r["token_count"])
        y = float(r["median_time_ns"]) / 1e6
        if x > 0 and y > 0 and x <= xmax:
            pts.append((x, y))
    return sorted(pts)


def fit_loglog(pts: list[tuple]) -> tuple[float, float] | None:
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
    if coeff <= 0 or a <= 0:
        return data_xmax
    try:
        x_at_ymax = (ymax / coeff) ** (1.0 / a)
    except (ZeroDivisionError, ValueError):
        return data_xmax
    return min(data_xmax, x_at_ymax)

# ---------------------------------------------------------------------------
# Panel generation (returns tikzpicture lines, no figure wrapper)
# ---------------------------------------------------------------------------

def write_panel(title: str, series: list, add_legend: bool) -> list[str]:
    # Compute data-driven x limits across all series
    all_x: list[float] = []
    for _, fname, *_ in series:
        pts = get_xy(load_csv(fname))
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

    for display_name, fname, color, mark, line_style in series:
        pts = get_xy(load_csv(fname))

        if not pts:
            lines.append(f"% no BRNGLR data for {display_name}")
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
                    f"% {title} BRNGLR {display_name}: n^{a:.2f}",
                ]

    lines += [r"\end{axis}", r"\end{tikzpicture}"]
    return lines

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    out: list[str] = [
        "% Generated by make_brnglr_grammar_comparison.py — do not edit by hand.",
        "% Caption and label live in the main .tex file.",
    ]

    panels = [
        ("JSON", SERIES_JSON, True),
        ("Expr", SERIES_EXPR, True),
    ]

    # Side-by-side minipages
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

    print("\nFitted power-law exponents (BRNGLR):")
    for title, series, _ in panels:
        print(f"  {title}:")
        for display_name, fname, *_ in series:
            pts = get_xy(load_csv(fname))
            fit = fit_loglog(pts)
            if fit:
                a, coeff = fit
                print(f"    {display_name:20s}  n^{a:.2f}  "
                      f"(n={pts[0][0]:.0f}..{pts[-1][0]:.0f}, {len(pts)} pts)")
            else:
                print(f"    {display_name:20s}  insufficient data ({len(pts)} pts)")


if __name__ == "__main__":
    main()
