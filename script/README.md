# Package artifact generation

These are the package copies of the scripts in `manuscript/bin/`, with
snake_case filenames and imports. They read benchmark datasets from `results/`
and grammar definitions from `grammars/` at the repository root. They write
publication TeX fragments to `manuscript/img/`, machine-readable summaries to
`results/analysis/`, and combined valid/invalid CSVs to `results_comprehensive/`.
Paths are resolved relative to the scripts, independently of the current
working directory, by `artifact_paths.py`.

From the repository root, regenerate every data-derived artifact imported by
the manuscript using the package datasets:

```sh
python3 script/generate_all.py
```

All generators use only the Python standard library.

The original scripts and frozen data remain in `manuscript/bin/` and
`manuscript/result/` for standalone manuscript builds. To regenerate using
those frozen inputs, run `make -C manuscript figures`. The manuscript's
individual `make` targets are named after each figure or table's `\label{}`,
with `:` replaced by `_` (e.g. `make -C manuscript tab_cykValiant`).

Dataset directories retain the benchmark-driver names used by the source
repository:

- `benchmark_csv/`: controlled valid-input experiments
- `benchmark_csv_invalid/`: controlled invalid-input experiments
- `benchmark_ws/`: whitespace-aware source-code experiments
- `benchmark_lex/`: lexer-first source-code experiments
- `benchmark_rlc/`: Reference Language Corpora experiments
- `benchmark_tree_sitter_java/`: Java comparison
- `benchmark_tree_sitter_stress/`: Tree-sitter stress grammars

The generated TeX preserves the manuscript's existing table and plot layout.
Rows with successful status but failed recognition or reconstruction are
excluded from valid-input summaries. Invalid-input summaries require confirmed
rejection. Timeouts and parse failures are never treated as measured zeroes.
