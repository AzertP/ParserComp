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
* Rendering mirrors make_runtime_table.py: decimal points are aligned
  per column via \\hphantom padding, and \\best/\\worst highlighting is
  based on the *displayed* (rounded) value, skipped entirely when every
  present parser rounds to the same number.
"""

import os
import csv
import statistics
from collections import defaultdict, OrderedDict

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
RESULTS_DIR  = os.path.join(PROJECT_ROOT, "results")

BUCKETS = [
    ("0–10",    0,   10),
    ("10–20",  10,   20),
    ("20–100", 20,  100),
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

# Display name -> the \newcommand macro that should appear in the table
# instead of the literal string (defined once in the main .tex file so
# a grammar's display name can be tweaked in one place).
GRAMMAR_MACROS = {
    "S-Expression": r"\GSExpression",
    "Expr (lr)":    r"\GExprLR",
    "Bool":         r"\GBool",
    "Expr (ambig)": r"\GExprAmbig",
    "Expr (rr)":    r"\GExprRR",
}

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


EM = "—"   # em-dash → rendered as --- in LaTeX


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
        return r"\multicolumn{1}{c|}{---}"
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
# raws_out carries the rounded displayed value per column, used for
# best/worst highlighting (see best_worst() above).
# ---------------------------------------------------------------------------

rows_out = []
raws_out = []
for display, filename, group in GRAMMAR_SOURCES:
    for label, lo, hi in BUCKETS:
        bucket_data = raw[display][label]
        if not any(bucket_data.get(p) for p in OUTPUT_COLUMNS):
            continue
        row = {"Grammar": display, "Group": group, "Bucket": label}
        raw_meds = {}
        for col in OUTPUT_COLUMNS:
            vals = bucket_data.get(col, [])
            formatted = fmt(vals)
            row[col + "_ms"] = formatted
            if formatted == EM:
                raw_meds[col] = None
            else:
                main, _ = main_and_suffix(formatted)
                raw_meds[col] = float(main)
        rows_out.append(row)
        raws_out.append(raw_meds)

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

def write_tex_table(rows, raws, out_tex):
    # Group rows (and their parallel raw-median dicts) by grammar,
    # preserving first-seen order.
    grammar_rows = OrderedDict()
    grammar_raws = OrderedDict()
    for r, rw in zip(rows, raws):
        grammar_rows.setdefault(r["Grammar"], []).append(r)
        grammar_raws.setdefault(r["Grammar"], []).append(rw)

    # Per-column max integer/fractional digit widths, computed across
    # every present cell in the whole table, so decimal points line up
    # vertically within a column.
    col_max_int = {c: 0 for c in OUTPUT_COLUMNS}
    col_max_frac = {c: 0 for c in OUTPUT_COLUMNS}
    for r in rows:
        for c in OUTPUT_COLUMNS:
            val = r[c + "_ms"]
            if val == EM:
                continue
            main, _ = main_and_suffix(val)
            ip, fp = split_num(main)
            col_max_int[c] = max(col_max_int[c], len(ip))
            col_max_frac[c] = max(col_max_frac[c], len(fp))

    # Bucket labels are also padded to a common width purely for a
    # readable, vertically-aligned .tex source.
    bucket_width = max(len(label) for label, _, _ in BUCKETS)

    lines = [
        r"\small",
        r"% Generated by make_cyk_valiant_table.py — do not edit by hand.",
        r"% Caption and label live in the main .tex file.",
        r"\begin{tabular}{|l|l|l|l|l|}",
        r"  \hline",
        r"  \textbf{Grammar} & \textbf{Tokens} & \textbf{CYK (\millisecond)}"
        r" & \textbf{Valiant (\millisecond)} & \textbf{Earley (\millisecond)} \\",
        r"  \hline\hline",
    ]

    grammar_list = list(grammar_rows.keys())
    for gi, grammar in enumerate(grammar_list):
        g_rows = grammar_rows[grammar]
        g_raws = grammar_raws[grammar]
        n = len(g_rows)
        grammar_macro = GRAMMAR_MACROS.get(grammar, grammar)
        for ri, (r, rw) in enumerate(zip(g_rows, g_raws)):
            gram_cell = rf"\multirow{{{n}}}{{*}}{{{grammar_macro}}}" if ri == 0 else ""
            bv, wv = best_worst(rw)
            cells = []
            for c in OUTPUT_COLUMNS:
                val = r[c + "_ms"]
                rv = rw.get(c)
                is_b = rv is not None and bv is not None and rv == bv
                is_w = rv is not None and wv is not None and rv == wv
                cells.append(
                    render_cell(val, col_max_int[c], col_max_frac[c], is_b, is_w)
                )
            bucket = r["Bucket"].ljust(bucket_width)
            lines.append(
                f"  {gram_cell} & {bucket} & " + " & ".join(cells) + r" \\"
            )
        lines.append(r"  \hline")

    lines.append(r"\end{tabular}")

    os.makedirs(os.path.dirname(out_tex), exist_ok=True)
    with open(out_tex, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"LaTeX table written to: {out_tex}")


OUTPUT_TEX = os.path.join(PROJECT_ROOT, "img", "cykValiantTable.tex")
write_tex_table(rows_out, raws_out, OUTPUT_TEX)
