#!/usr/bin/env python3
"""Regenerate every data-derived TeX artifact imported by the manuscript."""

import subprocess
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent

SCRIPTS = [
    "combine_benchmark_csv.py",
    "make_grammar_table.py",
    "make_input_counts_table.py",
    "make_cyk_valiant_table.py",
    "make_runtime_table.py",
    "make_memory_table.py",
    "make_lr_baseline_plot.py",
    "make_grammar_comparison_plot.py",
    "make_ll1_grammar_parser_comparison.py",
    "make_cyk_valiant_plot.py",
    "make_cyk_valiant_earley_plot.py",
    "make_all_grammars_general_plot.py",
    "make_brnglr_grammar_comparison.py",
    "make_rq4_external_table.py",
    "make_invalid_table.py",
    "make_treesitter_table.py",
]


def main():
    for script in SCRIPTS:
        print(f"==> {script}", flush=True)
        subprocess.run(
            [sys.executable, str(SCRIPT_DIR / script)],
            cwd=SCRIPT_DIR.parent,
            check=True,
        )


if __name__ == "__main__":
    main()
