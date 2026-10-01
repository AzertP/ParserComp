# Package artifact generation

These scripts read benchmark datasets from `results/` and grammar definitions
from `grammars/` at the repository root. They write publication TeX fragments
to `results/tex/`, machine-readable summaries to `results/analysis/`, and
combined valid/invalid CSVs to `results_comprehensive/`.
Paths are resolved relative to the scripts, independently of the current
working directory, by `artifact_paths.py`.

From the repository root, regenerate the tables, plots, and summaries from
the bundled datasets:

```sh
python3 script/generate_all.py
```

All generators use only the Python standard library.

The command creates the output directories as needed and replaces existing
generated files. To generate an individual table or plot, run its script
directly, for example `python3 script/make_grammar_table.py`.

The `.tex` outputs are fragments for inclusion in a LaTeX document. Rendering
them to PDF requires a document wrapper and a LaTeX installation; this command
generates the fragments and CSV summaries only.

Dataset directories retain the benchmark-driver names used by the source
repository:

- `benchmark_csv/`: controlled valid-input experiments
- `benchmark_csv_invalid/`: controlled invalid-input experiments
- `benchmark_ws/`: whitespace-aware source-code experiments
- `benchmark_lex/`: lexer-first source-code experiments
- `benchmark_rlc/`: Reference Language Corpora experiments
- `benchmark_tree_sitter_java/`: Java comparison
- `benchmark_tree_sitter_stress/`: Tree-sitter stress grammars

Rows with successful status but failed recognition or reconstruction are
excluded from valid-input summaries. Invalid-input summaries require confirmed
rejection. Timeouts and parse failures are never treated as measured zeroes.
