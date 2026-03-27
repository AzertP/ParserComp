#!/usr/bin/env python3
"""
grammar_stats.py
----------------
Read a grammar JSON file and report:
  - number of productions (total alternatives across all rules)
  - number of nonterminals
  - number of terminals
  - number of nullable nonterminals
  - whether any left recursion is present (direct or indirect)

Usage:
    python3 grammar_stats.py path/to/grammar.json [path/to/other.json ...]

Grammar JSON format (as used in this project):
    {
      "name": "...",
      "start": "<start_symbol>",
      "rules": {
        "<A>": [ ["sym1", "sym2", ...], [], ["sym3"], ... ],
        ...
      }
    }
Each key in "rules" is a nonterminal; each inner list is one alternative
(an empty list [] represents an epsilon production).
"""

import json
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Core analysis
# ---------------------------------------------------------------------------

def is_nonterminal(symbol: str, rules: dict) -> bool:
    return symbol in rules


def compute_nullable(rules: dict) -> set:
    """
    Fixed-point computation of the set of nullable nonterminals.
    A nonterminal N is nullable if:
      - it has an empty production [], OR
      - it has a production where every symbol is itself nullable.
    """
    nullable = set()

    # Seed: nonterminals with an explicit epsilon production
    for nt, alts in rules.items():
        if [] in alts:
            nullable.add(nt)

    # Iterate until stable
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


def compute_left_reachable(rules: dict, nullable: set) -> dict:
    """
    Build the left-reachability graph.
    There is an edge A -> B if B can appear as the first *effective* symbol
    of some production of A, i.e. there exists a production
      A -> α₁ α₂ … αₙ
    where α₁ … αₖ₋₁ are all nullable nonterminals and αₖ = B (a nonterminal).

    Returns a dict mapping each nonterminal to the set of nonterminals
    directly left-reachable from it in one step.
    """
    graph: dict[str, set] = {nt: set() for nt in rules}

    for nt, alts in rules.items():
        for alt in alts:
            for sym in alt:
                if is_nonterminal(sym, rules):
                    graph[nt].add(sym)
                # Stop extending past the first non-nullable symbol
                if sym not in nullable:
                    break

    return graph


def has_left_recursion(rules: dict, nullable: set) -> bool:
    """
    Return True if any nonterminal can reach itself via the left-reachability
    graph (i.e. the grammar contains direct or indirect left recursion).
    Uses iterative DFS with a visited/stack set.
    """
    graph = compute_left_reachable(rules, nullable)

    for start in rules:
        # DFS from `start`; detect if we ever return to `start`
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


def left_recursive_nonterminals(rules: dict, nullable: set) -> set:
    """
    Return the set of nonterminals that participate in left recursion
    (useful for debugging / reporting).
    """
    graph = compute_left_reachable(rules, nullable)
    recursive = set()

    for start in rules:
        visited: set = set()
        stack = list(graph[start])
        while stack:
            node = stack.pop()
            if node == start:
                recursive.add(start)
                break
            if node not in visited:
                visited.add(node)
                stack.extend(graph.get(node, []))

    return recursive


# ---------------------------------------------------------------------------
# Grammar metadata: display name + determinism classification
# Keys are the file stem (Path(path).stem).
# ll: True  → grammar is LL(1)   (implies LR(1) as well)
# lr: True  → grammar is LR(1) but not LL(1)
# ---------------------------------------------------------------------------

GRAMMAR_META: dict[str, dict] = {
    "ansi_c":           {"display": "ANSI C",            "ll": False, "lr": False},
    "bool":             {"display": "Bool",               "ll": False, "lr": False},
    "calc":             {"display": "Calculator",         "ll": False, "lr": True },
    "ll1_calc":         {"display": "Calculator LL(1)",   "ll": True,  "lr": True },
    "expr":             {"display": "Expr",               "ll": False, "lr": True },
    "expr_ambi":        {"display": "Expr Ambiguous",     "ll": False, "lr": False},
    "cpp":              {"display": "C++",                "ll": False, "lr": False},
    "css":              {"display": "CSS",                "ll": False, "lr": False},
    "html":             {"display": "HTML",               "ll": False, "lr": False},
    "json":             {"display": "JSON",               "ll": False, "lr": False},
    "ll1_json":         {"display": "JSON LL(1)",         "ll": True,  "lr": True },
    "lr_json":          {"display": "JSON LR(1)",         "ll": False, "lr": True },
    "jsl18":            {"display": "Java",               "ll": False, "lr": False},
    "pascal":           {"display": "Pascal",             "ll": False, "lr": False},
    "ll1_tinypascal":   {"display": "TinyPascal LL(1)",   "ll": True,  "lr": True },
    "sexp":             {"display": "S-Expression",       "ll": False, "lr": False},
    "ll1_sexp":         {"display": "S-Expression LL(1)", "ll": True,  "lr": True },
    "shell":            {"display": "Shell",              "ll": False, "lr": False},
    "sql":              {"display": "SQL",                "ll": False, "lr": False},
    "tinyc":            {"display": "TinyC",              "ll": False, "lr": False},
    "lr_tinyc":         {"display": "TinyC LR(1)",        "ll": False, "lr": True },
    # extras present in the directory
    "json_tokenized":   {"display": "JSON (tokenized)",   "ll": False, "lr": False},
    "simple":           {"display": "Simple",             "ll": False, "lr": False},
}


# ---------------------------------------------------------------------------
# Per-grammar stats
# ---------------------------------------------------------------------------

def analyse(path: str) -> dict:
    with open(path) as fh:
        g = json.load(fh)

    rules: dict = g["rules"]

    # Productions: total number of alternatives
    n_productions = sum(len(alts) for alts in rules.values())

    # Nonterminals
    nonterminals = set(rules.keys())
    n_nonterminals = len(nonterminals)

    # Terminals: symbols that appear in production bodies but are not nonterminals
    terminals: set = set()
    for alts in rules.values():
        for alt in alts:
            for sym in alt:
                if sym not in nonterminals:
                    terminals.add(sym)
    n_terminals = len(terminals)

    # Nullable nonterminals
    nullable = compute_nullable(rules)
    n_nullable = len(nullable)

    # Left recursion
    left_rec = has_left_recursion(rules, nullable)
    lr_nts   = left_recursive_nonterminals(rules, nullable) if left_rec else set()

    stem = Path(path).stem
    meta = GRAMMAR_META.get(stem, {})
    display_name = meta.get("display", g.get("name", stem))
    is_ll = meta.get("ll", False)
    is_lr = meta.get("lr", False) or is_ll   # LL(1) ⊆ LR(1)

    return {
        "name":           display_name,
        "stem":           stem,
        "start":          g.get("start", ""),
        "productions":    n_productions,
        "nonterminals":   n_nonterminals,
        "terminals":      n_terminals,
        "nullable":       n_nullable,
        "left_recursion": left_rec,
        "lr_nonterminals": sorted(lr_nts),
        "is_ll":          is_ll,
        "is_lr":          is_lr,
    }


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def print_stats(stats: dict, verbose: bool = False) -> None:
    print(f"Grammar      : {stats['name']}")
    print(f"Start        : {stats['start']}")
    print(f"Productions  : {stats['productions']}")
    print(f"Nonterminals : {stats['nonterminals']}")
    print(f"Terminals    : {stats['terminals']}")
    print(f"Nullable NTs : {stats['nullable']}")
    print(f"Left recursion: {'yes' if stats['left_recursion'] else 'no'}")
    if verbose and stats['lr_nonterminals']:
        print(f"  Left-recursive NTs ({len(stats['lr_nonterminals'])}):")
        for nt in stats['lr_nonterminals']:
            print(f"    {nt}")
    print()


def print_table(all_stats: list[dict]) -> None:
    """Print all grammars as a compact aligned table."""
    header = (f"{'Grammar':<24} {'Prod':>6} {'NT':>5} {'Term':>6}"
              f" {'Null':>6} {'LeftRec':>8} {'LL':>6} {'LR':>6}")
    print(header)
    print("-" * len(header))
    for s in all_stats:
        ll_str = "LL(1)" if s["is_ll"] else "—"
        lr_str = "LR(1)" if s["is_lr"] else "—"
        print(
            f"{s['name']:<24} {s['productions']:>6} {s['nonterminals']:>5}"
            f" {s['terminals']:>6} {s['nullable']:>6}"
            f" {'yes' if s['left_recursion'] else 'no':>8}"
            f" {ll_str:>6} {lr_str:>6}"
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    args = sys.argv[1:]
    if not args:
        print("Usage: grammar_stats.py [-v] <grammar.json> [...]", file=sys.stderr)
        sys.exit(1)

    verbose = "-v" in args
    paths = [a for a in args if not a.startswith("-")]

    all_stats = []
    for path in paths:
        try:
            s = analyse(path)
            all_stats.append(s)
        except Exception as e:
            print(f"ERROR processing {path}: {e}", file=sys.stderr)

    if len(all_stats) == 1:
        print_stats(all_stats[0], verbose=verbose)
    else:
        print_table(all_stats)
        if verbose:
            print()
            for s in all_stats:
                print_stats(s, verbose=True)


if __name__ == "__main__":
    main()
