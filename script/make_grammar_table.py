#!/usr/bin/env python3
"""
make_grammar_table.py
---------------------
Generate img/grammarTable.tex — the benchmark grammar properties table
(tab:grammars) for the paper.

For each benchmark grammar the table reports:
  Grammar, Category, Productions, NT, Terminals, Nullable, Left Rec,
  Ambiguous, LL, LR

Grammar stats (productions, NT, terminals, nullable, left recursion) are
computed directly from the JSON grammar files under grammars/.
Remaining columns (category, ambiguity, LL/LR classification) are encoded
in BENCHMARK_GRAMMARS below.

Usage:
    python3 bin/make_grammar_table.py

Output:
    img/grammarTable.tex
"""

import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
GRAMMARS_DIR = os.path.join(PROJECT_ROOT, "grammars")
OUTPUT_FILE  = os.path.join(PROJECT_ROOT, "img", "grammarTable.tex")

# ---------------------------------------------------------------------------
# Benchmark grammar registry
# (display_name, file_stem, category, ambiguous, is_ll, is_lr)
# is_ll=True  → grammar is LL(1) (implies LR(1))
# is_lr=True  → grammar is LR(1) but not LL(1)
# ambiguous: True / False / None (None → rendered as '?')
# ---------------------------------------------------------------------------

BENCHMARK_GRAMMARS = [
    # display name          stem               category   ambi   ll      lr
    # ---- LL(1) grammars (also LR(1)) ----
    ("S-Expr LL-1",         "ll1_sexp",        "Simple",  False, True,   True ),
    ("Expr LL-1",           "ll1_calc",        "Simple",  False, True,   True ),
    ("JSON LL-1",           "ll1_json",        "Moderate",False, True,   True ),
    ("TinyPascal",          "ll1_tinypascal",  "Moderate",False, True,   True ),
    # ---- LR(1) but not LL(1) ----
    ("Expr (lr)",           "calc",            "Simple",  False, False,  True ),
    ("Expr (rr)",           "expr",            "Simple",  False, False,  True ),
    ("JSON (rr)",           "json",            "Moderate",False, False,  True ),
    ("JSON (lr)",           "lr_json",         "Moderate",False, False,  True ),
    ("TinyC LR-1",         "lr_tinyc",        "Moderate",False, False,  True ),
    # ---- General context-free (neither LL(1) nor LR(1)) ----
    ("S-Expression",        "sexp",            "Simple",  True,  False,  False),
    ("Bool",                "bool",            "Simple",  True,  False,  False),
    ("Expr (ambig)",           "expr_ambi",       "Simple",  True,  False,  False),
    ("JSON (ambig)",           "json_ambi",       "Moderate",True,  False,  False),
    ("TinyC",               "tinyc",           "Moderate",True,  False,  False),
    ("ANSI~C",              "ansi_c",          "Complex", True,  False,  False),
    ("Pascal",              "pascal",          "Complex", True,  False,  False),
    ("Java",                "jsl18",           "Complex", True,  False,  False),
    ("C++",                 "cpp",             "Complex", True,  False,  False),
    ("CSS",                 "css",             "Complex", True,  False,  False),
    ("HTML",                "html",            "Complex", True,  False,  False),
    ("Shell",               "shell",           "Complex", True,  False,  False),
    ("SQL",                 "sql",             "Complex", True,  False,  False),
]

# ---------------------------------------------------------------------------
# Grammar analysis (inlined from grammar_stats.py)
# ---------------------------------------------------------------------------

def compute_nullable(rules: dict) -> set:
    nullable = set()
    for nt, alts in rules.items():
        if [] in alts:
            nullable.add(nt)
    changed = True
    while changed:
        changed = False
        for nt, alts in rules.items():
            if nt in nullable:
                continue
            for alt in alts:
                if all(sym in nullable for sym in alt):
                    nullable.add(nt)
                    changed = True
                    break
    return nullable


def has_left_recursion(rules: dict, nullable: set) -> bool:
    graph: dict[str, set] = {nt: set() for nt in rules}
    for nt, alts in rules.items():
        for alt in alts:
            for sym in alt:
                if sym in rules:
                    graph[nt].add(sym)
                if sym not in nullable:
                    break
    for start in rules:
        visited: set = set()
        stack = list(graph[start])
        while stack:
            node = stack.pop()
            if node == start:
                return True
            if node not in visited:
                visited.add(node)
                stack.extend(graph.get(node, []))
    return False


def analyse(stem: str) -> dict:
    path = os.path.join(GRAMMARS_DIR, f"{stem}.json")
    with open(path) as fh:
        g = json.load(fh)
    rules = g["rules"]
    nonterminals = set(rules.keys())
    terminals: set = set()
    for alts in rules.values():
        for alt in alts:
            for sym in alt:
                if sym not in nonterminals:
                    terminals.add(sym)
    nullable = compute_nullable(rules)
    left_rec = has_left_recursion(rules, nullable)
    return {
        "productions":    sum(len(a) for a in rules.values()),
        "nonterminals":   len(nonterminals),
        "terminals":      len(terminals),
        "nullable":       len(nullable),
        "left_recursion": left_rec,
    }

# ---------------------------------------------------------------------------
# LaTeX helpers
# ---------------------------------------------------------------------------

YES = r"\checkmark"
NO  = r"---"
UNK = r"?"


def bool_cell(val) -> str:
    if val is True:
        return YES
    if val is False:
        return NO
    return UNK   # None → ?


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parser_group(is_ll, is_lr):
    if is_ll:           return "ll1"
    if is_lr:           return "lr1"
    return "general"


# Group key -> the exact rotated label text. All rotated group labels
# are set \small so they fit comfortably within their (sometimes short)
# group block.
GROUP_LABELS = {
    "ll1":     r"\small LL(1)",
    "lr1":     r"\small LR(1)",
    "general": r"\small General context-free",
}

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


def main() -> None:
    # First pass: compute every row's cell values so column widths can be
    # aligned (padded) consistently across the whole table.
    rows = []   # list of (grp, display, cat, prod, nt, term, null, leftrec, ambi, ll, lr)
    for display, stem, cat, ambi, is_ll, is_lr in BENCHMARK_GRAMMARS:
        grp = parser_group(is_ll, is_lr)
        try:
            s = analyse(stem)
        except FileNotFoundError:
            print(f"WARNING: grammars/{stem}.json not found — skipping",
                  file=sys.stderr)
            continue
        lr_cell = YES if (is_lr or is_ll) else NO   # LL(1) ⊆ LR(1)
        ll_cell = YES if is_ll else NO
        display_macro = GRAMMAR_MACROS.get(display, display)
        rows.append((
            grp, display_macro, cat,
            str(s["productions"]), str(s["nonterminals"]),
            str(s["terminals"]), str(s["nullable"]),
            bool_cell(s["left_recursion"]), bool_cell(ambi),
            ll_cell, lr_cell,
        ))

    # Column-wise padding widths (all but the last column get right-padded
    # with spaces to the widest value in that column, purely for a
    # readable, vertically-aligned .tex source).
    ncols = 10   # display .. lr, i.e. all columns after grp
    widths = [0] * ncols
    for r in rows:
        for i, val in enumerate(r[1:]):
            widths[i] = max(widths[i], len(val))

    def pad_row(r):
        vals = list(r[1:])
        out = []
        for i, val in enumerate(vals):
            if i == len(vals) - 1:
                out.append(val)
            else:
                out.append(val.ljust(widths[i]))
        return out

    lines = [
        r"% Generated by make_grammar_table.py — do not edit by hand.",
        r"% Caption and label live in the main .tex file.",
        r"\begin{tabular}{|p{1.5em}|l|l|r|r|r|r|c|c|c|c|}",
        r"  \hline",
        r"  & \textbf{Grammar} & \textbf{Category}"
        r" & \textbf{Prod.} & \textbf{NT} & \textbf{Term.} & \textbf{Null.}"
        r" & \textbf{Left Rec.} & \textbf{Ambig} & \textbf{LL(1)} & \textbf{LR(1)} \\",
        r"  \hline\hline",
    ]

    group_order = []
    for r in rows:
        if r[0] not in group_order:
            group_order.append(r[0])
    group_counts = {g: sum(1 for r in rows if r[0] == g) for g in group_order}

    plain_labels = {
        "ll1":     "LL(1)",
        "lr1":     "LR(1)",
        "general": "General context-free",
    }

    prev_group = None
    for i, r in enumerate(rows):
        grp = r[0]
        if grp != prev_group:
            lines.append(
                f"  % ======================= {plain_labels[grp]}: "
                f"{group_counts[grp]} rows ======================="
            )
            lines.append(
                rf"  \multirow{{{group_counts[grp]}}}{{*}}"
                rf"{{\rotatebox{{90}}{{{GROUP_LABELS[grp]}}}}}"
            )
            prev_group = grp

        cells = pad_row(r)
        lines.append("   & " + " & ".join(cells) + r" \\")

        is_last_in_group = (i == len(rows) - 1) or (rows[i + 1][0] != grp)
        if is_last_in_group:
            lines.append(r"  \hline")

    lines.append(r"\end{tabular}")

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"Written: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
