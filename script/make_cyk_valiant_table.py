"""
make_cyk_valiant_table.py
-------------------------
Produce a CYK vs Valiant runtime table by grammar and small input-size bucket.

Buckets (token_count):  0–10,  10–20,  20–100

Columns: Grammar, Tokens, CYK_ms, Valiant_ms, Earley_ms

Notes
-----
* Only the base grammar variant is used for each grammar family.
  LL/LR-refactored variants (sexp_ll1, calc_ll1, …) are omitted because
  CYK and Valiant operate on CNF-converted grammars and their performance
  is determined by grammar structure, not LL/LR classification.
* Rows with status != "OK" are excluded.
* Grammars grouped by complexity (Simple / Moderate / Complex).
"""

import os
import csv
import statistics
from collections import defaultdict, OrderedDict

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
RESULTS_DIR  = os.path.join(PROJECT_ROOT, "results")

BUCKETS = [
    ("0\u201310",    0,   10),
    ("10\u201320",  10,   20),
    ("20\u2013100", 20,  100),
]

# (display_name, csv_filename, group)
GRAMMAR_SOURCES = [
    # --- Simple ---
    ("S-Expression", "benchmark_sexp.csv",      "Simple"),
    ("Expr (lr)",    "benchmark_calc.csv",       "Simple"),
    ("Bool",         "benchmark_bool.csv",        "Simple"),
    ("Expr (ambig)", "benchmark_expr_ambi.csv",   "Simple"),
    ("Expr (rr)",    "benchmark_expr.csv",         "Simple"),
]

# CSV parser name -> output column name (Leo = Earley with Leo optimisation)
PARSER_MAP = {"CYK": "CYK", "Valiant": "Valiant", "Leo": "Earley"}
OUTPUT_COLUMNS = ["CYK", "Valiant", "Earley"]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def bucket_label(n):
    for label, lo, hi in BUCKETS:
        if lo <= n < hi:
            return label
    return None


def ns_to_ms(ns):
    return float(ns) / 1_000_000


EM = "\u2014"   # em-dash → rendered as --- in LaTeX


def fmt(values):
    if not values:
        return EM

    def dp_for(v):
        """Decimal places determined by the magnitude of v."""
        if v < 1:
            return 3
        elif v < 10:
            return 2
        elif v < 100:
            return 1
        else:
            return 0

    m = statistics.median(values)
    dp = dp_for(m)
    med_str = f"{m:.{dp}f}"
    if len(values) < 2:
        return med_str
    s = statistics.stdev(values)
    std_str = f"{s:.{dp}f}"
    return med_str + r"\,{\footnotesize$\pm$\," + std_str + "}"


# ---------------------------------------------------------------------------
# Accumulate: raw[grammar][bucket][parser] = [values...]
# ---------------------------------------------------------------------------

raw = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

for display, filename, group in GRAMMAR_SOURCES:
    path = os.path.join(RESULTS_DIR, filename)
    try:
        with open(path, newline="") as fh:
            rows = list(csv.DictReader(fh))
    except FileNotFoundError:
        print(f"WARNING: {filename} not found — skipping")
        continue
    for row in rows:
        if row.get("status", "OK") != "OK":
            continue
        parser_csv = row["parser"]
        if parser_csv not in PARSER_MAP:
            continue
        col = PARSER_MAP[parser_csv]
        n = int(row["token_count"])
        b = bucket_label(n)
        if b is None:
            continue
        raw[display][b][col].append(ns_to_ms(row["median_time_ns"]))

# ---------------------------------------------------------------------------
# Build output rows (skip grammar+bucket combos with no data at all)
# ---------------------------------------------------------------------------

rows_out = []
for display, filename, group in GRAMMAR_SOURCES:
    for label, lo, hi in BUCKETS:
        bucket_data = raw[display][label]
        if not any(bucket_data.get(p) for p in OUTPUT_COLUMNS):
            continue
        rows_out.append({
            "Grammar":    display,
            "Group":      group,
            "Bucket":     label,
            "CYK_ms":     fmt(bucket_data.get("CYK",    [])),
            "Valiant_ms": fmt(bucket_data.get("Valiant", [])),
            "Earley_ms":  fmt(bucket_data.get("Earley",  [])),
        })

# ---------------------------------------------------------------------------
# Print plain-text preview
# ---------------------------------------------------------------------------

HEADER = ["Grammar", "Bucket", "CYK_ms", "Valiant_ms", "Earley_ms"]
COL_W  = [14, 10, 10, 12, 10]
sep      = "+" + "+".join("-" * (w + 2) for w in COL_W) + "+"
head_row = "|" + "|".join(f" {h:<{w}} " for h, w in zip(HEADER, COL_W)) + "|"
print(sep)
print(head_row)
print(sep)
for r in rows_out:
    line = "|" + "|".join(
        f" {str(r[h]):<{w}} " for h, w in zip(HEADER, COL_W)
    ) + "|"
    print(line)
print(sep)

# ---------------------------------------------------------------------------
# Write LaTeX  ->  img/cykValiantTable.tex
# ---------------------------------------------------------------------------

def cell(val):
    return r"\multicolumn{1}{c}{---}" if val == EM else val


def write_tex_table(rows, out_tex):
    # Group rows by grammar
    grammar_rows = OrderedDict()
    grammar_group = {}
    for r in rows:
        grammar_rows.setdefault(r["Grammar"], []).append(r)
        grammar_group[r["Grammar"]] = r["Group"]

    lines = [
        r"\small",
        r"% Generated by make_cyk_valiant_table.py — do not edit by hand.",
        r"% Caption and label live in the main .tex file.",
        r"\begin{tabular}{ll r r r}",
        r"  \toprule",
        r"  \textbf{Grammar} & \textbf{Tokens}"
        r" & \textbf{CYK (ms)} & \textbf{Valiant (ms)} & \textbf{Earley (ms)} \\",
        r"  \midrule",
    ]

    prev_group = None
    grammar_list = list(grammar_rows.keys())
    for gi, grammar in enumerate(grammar_list):
        grp = grammar_group[grammar]
        if grp != prev_group:
            if prev_group is not None:
                lines.append(r"  \midrule")
            lines.append(
                r"  \multicolumn{5}{@{\quad}l}{\textit{" + grp + r"}} \\"
            )
            lines.append(r"  \midrule")
            prev_group = grp

        g_rows = grammar_rows[grammar]
        n = len(g_rows)
        for ri, r in enumerate(g_rows):
            gram_cell = rf"\multirow{{{n}}}{{*}}{{{grammar}}}" if ri == 0 else ""
            lines.append(
                f"  {gram_cell} & {r['Bucket']}"
                f" & {cell(r['CYK_ms'])} & {cell(r['Valiant_ms'])}"
                f" & {cell(r['Earley_ms'])} \\\\"
            )
        if gi < len(grammar_list) - 1:
            lines.append(r"  \midrule")

    lines += [r"  \bottomrule", r"\end{tabular}"]

    os.makedirs(os.path.dirname(out_tex), exist_ok=True)
    with open(out_tex, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"LaTeX table written to: {out_tex}")


OUTPUT_TEX = os.path.join(PROJECT_ROOT, "img", "cykValiantTable.tex")
write_tex_table(rows_out, OUTPUT_TEX)
