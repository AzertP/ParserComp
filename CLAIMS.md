# Claims and Reproducibility Map

This table maps each figure in the paper to the data and script that produces it.  All paths are relative to the repository root.  Start with [reproduce.sh](reproduce.sh) to re-generate everything end-to-end; see [INSTALL.md](INSTALL.md) for prerequisites.

## Result files produced by the benchmark

The benchmark binary (`cargo run --release --bin benchmark_csv`) reads from  
`grammars/` and `input/`, and writes one CSV per configuration to `results/`.

| Output CSV | Grammar | Input file | Parsers |
|---|---|---|---|
| `results/benchmark_json_small.csv` | `grammars/json.json` | `input/json_small.txt` | Earley, GLL, RNGLR, BRNGLR, Valiant, CYK |
| `results/benchmark_sexp_small.csv` | `grammars/sexp.json` | `input/sexp_small.txt` | Earley, GLL, RNGLR, BRNGLR, Valiant, CYK |
| `results/benchmark_calc_small.csv` | `grammars/calc.json` | `input/calc_small.txt` | Earley, GLL, RNGLR, BRNGLR, Valiant, CYK |
| `results/benchmark_tinyc_small.csv` | `grammars/tinyc.json` | `input/tinyc_small.txt` | Earley, GLL, RNGLR, BRNGLR, Valiant, CYK |
| `results/benchmark_ansi_c_small.csv` | `grammars/ansi_c.json` | `input/ansi_c_small.txt` | Earley, GLL, RNGLR, BRNGLR |
| `results/benchmark_cpp_small.csv` | `grammars/cpp.json` | `input/cpp_small.txt` | Earley, GLL, RNGLR, BRNGLR |
| `results/benchmark_pascal_small.csv` | `grammars/pascal.json` | `input/pascal_small.txt` | Earley, GLL, RNGLR, BRNGLR |
| `results/benchmark_java_small.csv` | `grammars/jsl18.json` | `input/java_small.txt` | Earley, GLL, RNGLR, BRNGLR |
| `results/benchmark_json_large.csv` | `grammars/json.json` | `input/json_large.txt` | Earley, GLL, RNGLR, BRNGLR |
| `results/benchmark_sexp_large.csv` | `grammars/sexp.json` | `input/sexp_large.txt` | Earley, GLL, RNGLR, BRNGLR |
| `results/benchmark_calc_large.csv` | `grammars/calc.json` | `input/calc_large.txt` | Earley, GLL, RNGLR, BRNGLR |
| `results/benchmark_tinyc_large.csv` | `grammars/tinyc.json` | `input/tinyc_large.txt` | Earley, GLL, RNGLR, BRNGLR |
| `results/benchmark_ansi_c_large.csv` | `grammars/ansi_c.json` | `input/ansi_c_large.txt` | Earley, GLL, RNGLR, BRNGLR |
| `results/benchmark_cpp_large.csv` | `grammars/cpp.json` | `input/cpp_large.txt` | Earley, GLL, RNGLR, BRNGLR |
| `results/benchmark_pascal_large.csv` | `grammars/pascal.json` | `input/pascal_large.txt` | Earley, GLL, RNGLR, BRNGLR |
| `results/benchmark_java_large.csv` | `grammars/jsl18.json` | `input/java_large.txt` | Earley, GLL, RNGLR, BRNGLR |
| `results/benchmark_calc_with_lllr_large.csv` | `grammars/ll1_calc.json` | `input/calc_large.txt` | Earley, GLL, RNGLR, BRNGLR, LL, LR |
| `results/benchmark_json_with_lllr_large.csv` | `grammars/ll1_json.json` | `input/json_large.txt` | GLL, RNGLR, BRNGLR, LL, LR |
| `results/benchmark_json_with_lllr_earley.csv` | `grammars/ll1_json.json` | `input/json_medium.txt` | Earley |
| `results/benchmark_sexp_with_lllr_large.csv` | `grammars/ll1_sexp.json` | `input/sexp_ll1.txt` | Earley, GLL, RNGLR, BRNGLR, LL, LR |
| `results/benchmark_tinypascal_with_lllr_large.csv` | `grammars/ll1_tinypascal.json` | `input/tinypascal_large.txt` | Earley, GLL, RNGLR, BRNGLR, LL, LR |
| `results/benchmark_lr_json_large.csv` | `grammars/lr_json.json` | `input/json_large.txt` | Earley, GLL, RNGLR, BRNGLR, LR |
| `results/benchmark_lr_tinyc_large.csv` | `grammars/lr_tinyc.json` | `input/tinyc_lr_large.txt` | Earley, GLL, RNGLR, BRNGLR, LR |
| `results/benchmark_json_earley.csv` | `grammars/json.json` | `input/json_medium.txt` | Earley |

## Figures produced by the plotter

Run `python3 plotter.py` (from the repository root) to regenerate all figures into `plot/`.

| Paper figure | Output file | Plot type | Input CSVs (from `results/`) | Parser set |
|---|---|---|---|---|
| Small input — time | `plot/small_cyk_valiant_time.pdf` | Scatter + trend per grammar | `benchmark_{grammar}_small.csv` × {json, sexp, calc, tinyc} | CYK, Valiant |
| Small input — memory | `plot/small_cyk_valiant_memory.pdf` | Scatter + trend per grammar | same | CYK, Valiant |
| Large input — time | `plot/large_general_time.pdf` | Scatter + trend per grammar | `benchmark_{grammar}_large.csv` × 8 grammars | Earley, GLL, RNGLR, BRNGLR |
| Large input — memory | `plot/large_general_memory.pdf` | Scatter + trend per grammar | same | Earley, GLL, RNGLR, BRNGLR |
| LL/LR baseline — time | `plot/ll_lr_baseline_time.pdf` | Scatter + trend per grammar | `benchmark_{grammar}_with_lllr_large.csv` × {calc, json, sexp, tinypascal} | Earley, GLL, RNGLR, BRNGLR, LL, LR |
| LL/LR baseline — memory | `plot/ll_lr_baseline_memory.pdf` | Scatter + trend per grammar | same | same |
| LR comparison — time | `plot/lr_comparison_time.pdf` | Scatter + trend per grammar | `benchmark_lr_{grammar}_large.csv` × {tinyc, json} | Earley, GLL, RNGLR, BRNGLR, LR, LL |
| LR comparison — memory | `plot/lr_comparison_memory.pdf` | Scatter + trend per grammar | same | same |
| Memory ceiling bar chart | `plot/memory_max_limit_*.pdf` | Bar chart (max memory cap) | `benchmark_{grammar}_large.csv` × 8 grammars | Earley, GLL, RNGLR, BRNGLR |

## CSV column schema

Each `results/benchmark_*.csv` has the following columns:

| Column | Type | Description |
|---|---|---|
| `parser` | string | Parser name (e.g., `Earley`, `GLL`, `RNGLR`, `BRNGLR`, `CYK`, `Valiant`, `LL`, `LR`) |
| `input_length` | int | Number of characters in the input |
| `token_count` | int | Number of scanned tokens |
| `median_time_ns` | float | Median parse time in nanoseconds across repeated runs |
| `mad_ns` | float | Median absolute deviation of parse times in nanoseconds |
| `peak_memory_bytes` | int | Approximate peak memory delta in bytes (sampled at 1 ms intervals) |
| `iterations` | int | Number of timed iterations performed |
| `success` | bool | Whether the parser produced a valid parse tree |
