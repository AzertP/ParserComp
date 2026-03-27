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

def main() -> None:
    lines = [
        r"% Generated by make_grammar_table.py — do not edit by hand.",
        r"% Caption and label live in the main .tex file.",
        r"\begin{tabular}{llrrrr ccccc}",
        r"  \toprule",
        r"  \textbf{Grammar} & \textbf{Category}"
        r" & \textbf{Prod.} & \textbf{NT} & \textbf{Term.} & \textbf{Null.}"
        r" & \textbf{Left Rec.} & \textbf{Ambiguous} & \textbf{LL(1)} & \textbf{LR(1)} \\",
        r"  \midrule",
    ]

    def parser_group(is_ll, is_lr):
        if is_ll:           return "ll1"
        if is_lr:           return "lr1"
        return "general"

    prev_group = None
    for display, stem, cat, ambi, is_ll, is_lr in BENCHMARK_GRAMMARS:
        grp = parser_group(is_ll, is_lr)
        # Insert a labelled group separator between parser-type sections
        if prev_group is not None and grp != prev_group:
            group_labels = {
                "ll1":     r"LL(1)",
                "lr1":     r"LR(1), not LL(1)",
                "general": r"General context-free",
            }
            lines.append(r"  \midrule")
            lines.append(
                r"  \multicolumn{10}{@{\quad}l}{\textit{"
                + group_labels[grp]
                + r"}} \\"
            )
            lines.append(r"  \midrule")
        prev_group = grp

        try:
            s = analyse(stem)
        except FileNotFoundError:
            print(f"WARNING: grammars/{stem}.json not found — skipping",
                  file=sys.stderr)
            continue

        lr_cell = YES if (is_lr or is_ll) else NO   # LL(1) ⊆ LR(1)
        ll_cell = YES if is_ll else NO

        lines.append(
            f"  {display} & {cat}"
            f" & {s['productions']} & {s['nonterminals']}"
            f" & {s['terminals']} & {s['nullable']}"
            f" & {bool_cell(s['left_recursion'])}"
            f" & {bool_cell(ambi)}"
            f" & {ll_cell} & {lr_cell} \\\\"
        )

    lines += [
        r"  \bottomrule",
        r"\end{tabular}",
    ]

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"Written: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
