"""
make_memory_table.py
--------------------
Produce Table T3: Peak Memory Summary by Bucket.

Buckets (token_count):
    0–5k, 5k–15k, 15k–30k

Columns:
    Grammar, Bucket, LL(1)_MB, LR(1)_MB, Earley_MB, GLL_MB, RNGLR_MB, BRNGLR_MB

Aggregation:
    Median of peak_memory_bytes (converted to MB) within each (grammar, bucket).
    Rows with status != "OK" (e.g. CONFLICT) are excluded.
    Rows with peak_memory_bytes == 0 are excluded (sub-threshold measurement
    artefacts; genuine zero-allocation inputs fall below OS page granularity).

Parser name mapping (CSV name -> column):
    Leo   -> Earley   (Earley with Leo right-recursion optimisation)
    LL    -> LL(1)
    LR    -> LR(1)

Notes on file selection
-----------------------
*  Each CSV is a distinct grammar variant and becomes its own table row.
   No data from different grammar files is ever merged into one row.
*  LL/LR columns show --- when the grammar is not LL(1)/LR(1).
*  Variants with an LL(1)-refactored grammar appear as separate rows
   (e.g. "JSON" vs "JSON LL-1").
"""

import os
import csv
import statistics
from collections import defaultdict, OrderedDict

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
RESULTS_DIR  = os.path.join(PROJECT_ROOT, "results")

BUCKETS = [
    ("0\u20135k",    0,       5_000),
    ("5k\u201315k",  5_000,  15_000),
    ("15k\u201330k", 15_000, 30_000),
]

# Grammar display name -> list of (file, parser_csv_name -> output_column).
# "Leo" maps to "Earley" (Leo = Earley + Leo right-recursion optimisation).
# Only rows with status == "OK" are used; CONFLICT rows are silently dropped.

GRAMMAR_SOURCES = OrderedDict([
    # ---- LL(1) grammars (all six parser columns populated) ----
    ("TinyPascal",   [("benchmark_tinypascal.csv",
                       {"LL": "LL(1)", "LR": "LR(1)", "Leo": "Earley",
                        "GLL": "GLL", "RNGLR": "RNGLR", "BRNGLR": "BRNGLR"})]),
    ("S-Expr LL-1",  [("benchmark_sexp_ll1.csv",
                       {"LL": "LL(1)", "LR": "LR(1)", "Leo": "Earley",
                        "GLL": "GLL", "RNGLR": "RNGLR", "BRNGLR": "BRNGLR"})]),
    ("Expr LL-1",    [("benchmark_calc_ll1.csv",
                       {"LL": "LL(1)", "LR": "LR(1)", "Leo": "Earley",
                        "GLL": "GLL", "RNGLR": "RNGLR", "BRNGLR": "BRNGLR"})]),
    ("JSON LL-1",    [("benchmark_json_ll1.csv",
                       {"LL": "LL(1)", "LR": "LR(1)", "Leo": "Earley",
                        "GLL": "GLL", "RNGLR": "RNGLR", "BRNGLR": "BRNGLR"})]),
    # ---- LR(1) but not LL(1) (LR column populated, LL is ---) ----
    ("JSON (rr)",    [("benchmark_json.csv",
                       {"LR": "LR(1)", "Leo": "Earley", "GLL": "GLL",
                        "RNGLR": "RNGLR", "BRNGLR": "BRNGLR"})]),
    ("JSON (lr)",    [("benchmark_json_lr.csv",
                       {"LR": "LR(1)", "Leo": "Earley", "GLL": "GLL",
                        "RNGLR": "RNGLR", "BRNGLR": "BRNGLR"})]),
    ("Expr (lr)",    [("benchmark_calc.csv",
                       {"LR": "LR(1)", "Leo": "Earley", "GLL": "GLL",
                        "RNGLR": "RNGLR", "BRNGLR": "BRNGLR"})]),
    ("Expr (rr)",    [("benchmark_expr.csv",
                       {"LR": "LR(1)", "Leo": "Earley", "GLL": "GLL",
                        "RNGLR": "RNGLR", "BRNGLR": "BRNGLR"})]),
    ("TinyC LR-1",   [("benchmark_tinyc_lr.csv",
                       {"LR": "LR(1)", "Leo": "Earley", "GLL": "GLL",
                        "RNGLR": "RNGLR", "BRNGLR": "BRNGLR"})]),
    # ---- General context-free (LL and LR both ---) ----
    ("S-Expression", [("benchmark_sexp.csv",
                       {"Leo": "Earley", "GLL": "GLL",
                        "RNGLR": "RNGLR", "BRNGLR": "BRNGLR"})]),
    ("TinyC",        [("benchmark_tinyc.csv",
                       {"Leo": "Earley", "GLL": "GLL",
                        "RNGLR": "RNGLR", "BRNGLR": "BRNGLR"})]),
    ("Bool",         [("benchmark_bool.csv",
                       {"Leo": "Earley", "GLL": "GLL",
                        "RNGLR": "RNGLR", "BRNGLR": "BRNGLR"})]),
    ("Expr (ambig)",    [("benchmark_expr_ambi.csv",
                       {"Leo": "Earley", "GLL": "GLL",
                        "RNGLR": "RNGLR", "BRNGLR": "BRNGLR"})]),
    ("JSON (ambig)",    [("benchmark_json_ambi.csv",
                       {"Leo": "Earley", "GLL": "GLL",
                        "RNGLR": "RNGLR", "BRNGLR": "BRNGLR"})]),
    ("ANSI C",       [("benchmark_ansi_c.csv",
                       {"Leo": "Earley", "GLL": "GLL",
                        "RNGLR": "RNGLR", "BRNGLR": "BRNGLR"})]),
    ("Pascal",       [("benchmark_pascal.csv",
                       {"Leo": "Earley", "GLL": "GLL",
                        "RNGLR": "RNGLR", "BRNGLR": "BRNGLR"})]),
    ("Java",         [("benchmark_java.csv",
                       {"Leo": "Earley", "GLL": "GLL",
                        "RNGLR": "RNGLR", "BRNGLR": "BRNGLR"})]),
    ("C++",          [("benchmark_cpp.csv",
                       {"Leo": "Earley", "GLL": "GLL",
                        "RNGLR": "RNGLR", "BRNGLR": "BRNGLR"})]),
    ("CSS",          [("benchmark_css.csv",
                       {"Leo": "Earley", "GLL": "GLL",
                        "RNGLR": "RNGLR", "BRNGLR": "BRNGLR"})]),
    ("HTML",         [("benchmark_html.csv",
                       {"Leo": "Earley", "GLL": "GLL",
                        "RNGLR": "RNGLR", "BRNGLR": "BRNGLR"})]),
    ("Shell",        [("benchmark_shell.csv",
                       {"Leo": "Earley", "GLL": "GLL",
                        "RNGLR": "RNGLR", "BRNGLR": "BRNGLR"})]),
    ("SQL",          [("benchmark_sql.csv",
                       {"Leo": "Earley", "GLL": "GLL",
                        "RNGLR": "RNGLR", "BRNGLR": "BRNGLR"})]),
])

OUTPUT_COLUMNS = ["LL(1)", "LR(1)", "Earley", "GLL", "RNGLR", "BRNGLR"]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def bucket_label(n):
    for label, lo, hi in BUCKETS:
        if lo <= n < hi:
            return label
    return None


def load_file(filename):
    path = os.path.join(RESULTS_DIR, filename)
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def bytes_to_mb(b):
    return float(b) / (1024 * 1024)


# ---------------------------------------------------------------------------
# Accumulate: raw[grammar][bucket][column] = [values...]
# Zero peak_memory_bytes are excluded (sub-threshold OS artefact).
# ---------------------------------------------------------------------------

raw = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

for grammar, sources in GRAMMAR_SOURCES.items():
    for filename, parser_map in sources:
        try:
            rows = load_file(filename)
        except FileNotFoundError:
            print(f"WARNING: {filename} not found — skipping")
            continue
        for row in rows:
            if row.get("status", "OK") != "OK":
                continue
            mem_bytes = float(row["peak_memory_bytes"])
            if mem_bytes == 0:
                continue   # sub-threshold artefact
            parser_csv = row["parser"]
            if parser_csv not in parser_map:
                continue
            col = parser_map[parser_csv]
            n = int(row["token_count"])
            b = bucket_label(n)
            if b is None:
                continue
            raw[grammar][b][col].append(bytes_to_mb(mem_bytes))

# ---------------------------------------------------------------------------
# Aggregate: median ± stdev within bucket
# ---------------------------------------------------------------------------

EM = "\u2014"   # em-dash placeholder; rendered as --- in LaTeX


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


rows_out = []
for grammar in GRAMMAR_SOURCES:
    for label, lo, hi in BUCKETS:
        bucket_data = raw[grammar][label]
        if not any(bucket_data.get(c) for c in OUTPUT_COLUMNS):
            continue
        row = {"Grammar": grammar, "Bucket": label}
        for col in OUTPUT_COLUMNS:
            row[col + "_MB"] = fmt(bucket_data.get(col, []))
        rows_out.append(row)

# ---------------------------------------------------------------------------
# Write CSV
# ---------------------------------------------------------------------------

HEADER = ["Grammar", "Bucket"] + [c + "_MB" for c in OUTPUT_COLUMNS]
out_csv = os.path.join(PROJECT_ROOT, "table_t3_memory_by_bucket.csv")
with open(out_csv, "w", newline="") as fh:
    writer = csv.DictWriter(fh, fieldnames=HEADER)
    writer.writeheader()
    writer.writerows(rows_out)
print(f"CSV written to: {out_csv}")

# ---------------------------------------------------------------------------
# Print plain-text preview
# ---------------------------------------------------------------------------

COL_W = [15, 10, 10, 10, 10, 10, 10, 10]


def pad(s, w):
    return str(s).ljust(w)


sep      = "+" + "+".join("-" * (w + 2) for w in COL_W) + "+"
head_row = "|" + "|".join(f" {pad(h, w)} " for h, w in zip(HEADER, COL_W)) + "|"
print(sep)
print(head_row)
print(sep)
for row in rows_out:
    line = "|" + "|".join(f" {pad(row[h], w)} " for h, w in zip(HEADER, COL_W)) + "|"
    print(line)
print(sep)

# ---------------------------------------------------------------------------
# Write LaTeX table  ->  img/memoryByBucket.tex
# ---------------------------------------------------------------------------

def cell(val):
    return r"\multicolumn{1}{c}{---}" if val == EM else val


# Grammar name -> parser-type group label (controls group separator rows)
GRAMMAR_GROUPS = OrderedDict([
    ("TinyPascal",   "LL(1)"),
    ("S-Expr LL-1",  "LL(1)"),
    ("Expr LL-1",    "LL(1)"),
    ("JSON LL-1",    "LL(1)"),
    ("JSON (rr)",    "LR(1), not LL(1)"),
    ("JSON (lr)",    "LR(1), not LL(1)"),
    ("Expr (lr)",    "LR(1), not LL(1)"),
    ("Expr (rr)",    "LR(1), not LL(1)"),
    ("TinyC LR-1",   "LR(1), not LL(1)"),
    ("S-Expression", "General context-free"),
    ("TinyC",        "General context-free"),
    ("Bool",         "General context-free"),
    ("Expr (ambig)",    "General context-free"),
    ("JSON (ambig)",    "General context-free"),
    ("ANSI C",       "General context-free"),
    ("Pascal",       "General context-free"),
    ("Java",         "General context-free"),
    ("C++",          "General context-free"),
    ("CSS",          "General context-free"),
    ("HTML",         "General context-free"),
    ("Shell",        "General context-free"),
    ("SQL",          "General context-free"),
])


def write_tex_table(rows, out_tex):
    groups = OrderedDict()
    for r in rows:
        groups.setdefault(r["Grammar"], []).append(r)

    lines = [
        r"\small",
        r"% Generated by make_memory_table.py — do not edit by hand.",
        r"% Caption and label live in the main .tex file.",
        r"\begin{tabular}{ll r r r r r r}",
        r"  \toprule",
        r"  \textbf{Grammar} & \textbf{Tokens}",
        r"    & \multicolumn{2}{c}{\textbf{Deterministic (MB)}}",
        r"    & \multicolumn{4}{c}{\textbf{Generalised (MB)}} \\",
        r"  \cmidrule(lr){3-4} \cmidrule(l){5-8}",
        r"  & & \textbf{LL(1)} & \textbf{LR(1)} & \textbf{Earley}"
        r" & \textbf{GLL} & \textbf{RNGLR} & \textbf{BRNGLR} \\",
        r"  \midrule",
    ]

    grammar_list = list(groups.keys())
    prev_grp = None
    for gi, grammar in enumerate(grammar_list):
        grp = GRAMMAR_GROUPS.get(grammar, "General context-free")
        if grp != prev_grp:
            if prev_grp is not None:
                lines.append(r"  \midrule")
            lines.append(
                r"  \multicolumn{8}{@{\quad}l}{\textit{" + grp + r"}} \\"
            )
            lines.append(r"  \midrule")
            prev_grp = grp

        g_rows = groups[grammar]
        n = len(g_rows)
        for ri, r in enumerate(g_rows):
            gram_cell = rf"\multirow{{{n}}}{{*}}{{{grammar}}}" if ri == 0 else ""
            lines.append(
                f"  {gram_cell} & {r['Bucket']}"
                f" & {cell(r['LL(1)_MB'])} & {cell(r['LR(1)_MB'])}"
                f" & {cell(r['Earley_MB'])} & {cell(r['GLL_MB'])}"
                f" & {cell(r['RNGLR_MB'])} & {cell(r['BRNGLR_MB'])} \\\\"
            )
        if gi < len(grammar_list) - 1:
            lines.append(r"  \midrule")

    lines += [r"  \bottomrule", r"\end{tabular}"]

    os.makedirs(os.path.dirname(out_tex), exist_ok=True)
    with open(out_tex, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"LaTeX table written to: {out_tex}")


out_tex = os.path.join(PROJECT_ROOT, "img", "memoryByBucket.tex")
write_tex_table(rows_out, out_tex)
