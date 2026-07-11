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


def main_and_suffix(formatted):
    """Split a cell's formatted string into the plain number and the
    trailing stderr suffix (which starts with '\\,{\\footnotesize')."""
    marker = r"\,{\footnotesize"
    if marker in formatted:
        idx = formatted.index(marker)
        return formatted[:idx], formatted[idx:]
    return formatted, ""


rows_out = []
raws_out = []
for grammar in GRAMMAR_SOURCES:
    for label, lo, hi in BUCKETS:
        bucket_data = raw[grammar][label]
        if not any(bucket_data.get(c) for c in OUTPUT_COLUMNS):
            continue
        row = {"Grammar": grammar, "Bucket": label}
        raw_meds = {}
        for col in OUTPUT_COLUMNS:
            vals = bucket_data.get(col, [])
            formatted = fmt(vals)
            row[col + "_MB"] = formatted
            if formatted == EM:
                raw_meds[col] = None
            else:
                # Best/worst is judged on the *displayed* (rounded) value,
                # not the unrounded raw median: if two parsers round to
                # the same printed number, there is no visible distinction
                # to highlight, even if their raw medians differ slightly.
                main, _ = main_and_suffix(formatted)
                raw_meds[col] = float(main)
        rows_out.append(row)
        raws_out.append(raw_meds)

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

# Grammar name -> parser-type group label (controls group separator rows)
GRAMMAR_GROUPS = OrderedDict([
    ("TinyPascal",   "LL(1)"),
    ("S-Expr LL-1",  "LL(1)"),
    ("Expr LL-1",    "LL(1)"),
    ("JSON LL-1",    "LL(1)"),
    ("JSON (rr)",    "LR(1)"),
    ("JSON (lr)",    "LR(1)"),
    ("Expr (lr)",    "LR(1)"),
    ("Expr (rr)",    "LR(1)"),
    ("TinyC LR-1",   "LR(1)"),
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

# Display name -> the \newcommand macro that should appear in the table
# instead of the literal string (defined once in the main .tex file so
# a grammar's display name can be tweaked in one place).
GRAMMAR_MACROS = {
    "TinyPascal":   r"\GTinyPascal",
    "S-Expr LL-1":  r"\GSExprLLA",
    "Expr LL-1":    r"\GExprLLA",
    "JSON LL-1":    r"\GJSONLLA",
    "JSON (rr)":    r"\GJSONRR",
    "JSON (lr)":    r"\GJSONLR",
    "Expr (lr)":    r"\GExprLR",
    "Expr (rr)":    r"\GExprRR",
    "TinyC LR-1":   r"\GTinyCLRA",
    "S-Expression": r"\GSExpression",
    "TinyC":        r"\GTinyC",
    "Bool":         r"\GBool",
    "Expr (ambig)": r"\GExprAmbig",
    "JSON (ambig)": r"\GJSONAmbig",
    "ANSI~C":       r"\GANSIC",
    "ANSI C":       r"\GANSIC",
    "Pascal":       r"\GPascal",
    "Java":         r"\GJava",
    "C++":          r"\GCPP",
    "CSS":          r"\GCSS",
    "HTML":         r"\GHTML",
    "Shell":        r"\GShell",
    "SQL":          r"\GSQL",
}


def split_num(s):
    """Split a plain formatted number like '15.3' or '356' into
    (integer_part, fractional_part). fractional_part is '' if the
    number has no decimal point."""
    if "." in s:
        i, f = s.split(".", 1)
        return i, f
    return s, ""


def best_worst(raw_meds):
    """Return (best_value, worst_value) among the present (non-None)
    displayed values in a row, or (None, None) if there is no visible
    distinction to highlight (no data present, or every present value
    rounds to the same displayed number)."""
    vals = [v for v in raw_meds.values() if v is not None]
    if not vals:
        return None, None
    bv, wv = min(vals), max(vals)
    if bv == wv:
        return None, None
    return bv, wv


def render_cell(formatted, max_int, max_frac, is_best, is_worst):
    if formatted == EM:
        return "---"
    main, suffix = main_and_suffix(formatted)
    ip, fp = split_num(main)
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
    body += suffix
    if is_best:
        return r"\best{" + body + "}"
    if is_worst:
        return r"\worst{" + body + "}"
    return body


def write_tex_table(rows, raws, out_tex):
    # Group rows (and their parallel raw-median dicts) by grammar,
    # preserving first-seen order.
    grammar_rows = OrderedDict()
    grammar_raws = OrderedDict()
    for r, rw in zip(rows, raws):
        grammar_rows.setdefault(r["Grammar"], []).append(r)
        grammar_raws.setdefault(r["Grammar"], []).append(rw)

    # Per-column max integer/fractional digit widths, computed across
    # every present cell in the whole table, so that decimal points
    # line up vertically within a column.
    col_max_int = {c: 0 for c in OUTPUT_COLUMNS}
    col_max_frac = {c: 0 for c in OUTPUT_COLUMNS}
    for r in rows:
        for c in OUTPUT_COLUMNS:
            val = r[c + "_MB"]
            if val == EM:
                continue
            main, _ = main_and_suffix(val)
            ip, fp = split_num(main)
            col_max_int[c] = max(col_max_int[c], len(ip))
            col_max_frac[c] = max(col_max_frac[c], len(fp))

    lines = [
        r"%\small",
        r"% Generated by make_memory_table.py — do not edit by hand.",
        r"% Caption and label live in the main .tex file.",
        r"\begin{tabular}{|p{1.5em}|l|l|l|l|l|l|l|l|}",
        r"  \hline",
        r"  & \multirow{2}{*}{\textbf{Grammar}} & \multirow{2}{*}{\textbf{Tokens}}",
        r"    & \multicolumn{2}{c|}{\textbf{Deterministic (MB)}}",
        r"    & \multicolumn{4}{c|}{\textbf{Generalised (MB)}} \\",
        r"  & & & \textbf{LL(1)} & \textbf{LR(1)} & \textbf{Earley}"
        r" & \textbf{GLL} & \textbf{RNGLR} & \textbf{BRNGLR} \\",
        r"  \hline\hline",
    ]

    grammar_list = list(grammar_rows.keys())
    prev_grp = None
    for gi, grammar in enumerate(grammar_list):
        grp = GRAMMAR_GROUPS.get(grammar, "General context-free")
        if grp != prev_grp:
            if prev_grp is not None:
                lines.append(r"  \hline")
            n_group_rows = sum(
                len(grammar_rows[g]) for g in grammar_list
                if GRAMMAR_GROUPS.get(g, "General context-free") == grp
            )
            lines.append(
                f"  % ======================= {grp}: "
                f"{n_group_rows} rows ======================="
            )
            lines.append(
                rf"  \multirow{{{n_group_rows}}}{{*}}"
                rf"{{\rotatebox{{90}}{{\small {grp}}}}}"
            )
            prev_grp = grp

        g_rows = grammar_rows[grammar]
        g_raws = grammar_raws[grammar]
        n = len(g_rows)
        grammar_macro = GRAMMAR_MACROS.get(grammar, grammar)
        for ri, (r, rw) in enumerate(zip(g_rows, g_raws)):
            gram_cell = rf"\multirow{{{n}}}{{*}}{{{grammar_macro}}}" if ri == 0 else ""
            bv, wv = best_worst(rw)
            cells = []
            for c in OUTPUT_COLUMNS:
                val = r[c + "_MB"]
                rv = rw.get(c)
                is_b = rv is not None and bv is not None and rv == bv
                is_w = (
                    rv is not None and wv is not None and rv == wv
                    and bv != wv
                )
                cells.append(
                    render_cell(val, col_max_int[c], col_max_frac[c], is_b, is_w)
                )
            lines.append(
                f"   & {gram_cell} & {r['Bucket']} & " + " & ".join(cells) + r" \\"
            )

        is_last_in_group = (
            gi == len(grammar_list) - 1
            or GRAMMAR_GROUPS.get(grammar_list[gi + 1], "General context-free") != grp
        )
        if not is_last_in_group:
            lines.append(r"  \cline{2-9}")

    lines += [r"  \hline", r"\end{tabular}"]

    os.makedirs(os.path.dirname(out_tex), exist_ok=True)
    with open(out_tex, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"LaTeX table written to: {out_tex}")


out_tex = os.path.join(PROJECT_ROOT, "img", "memoryByBucket.tex")
write_tex_table(rows_out, raws_out, out_tex)
