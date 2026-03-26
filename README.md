# Generalised Parser Comparison

Contains the full implementation of five generalised parsing algorithms (CYK, Valiant, Earley, GLL, RNGLR, BRNGLR) benchmarked against various context-free grammars, together with experiment results and input corpora.  

## Quick start

```bash
bash reproduce.sh
```

Or step by step:

```bash
cargo build --release
cargo run --release --bin benchmark_csv   # writes results/*.csv
```

See [INSTALL.md](INSTALL.md) for detailed prerequisite setup (Rust, Python, m4ri).

## Repository structure

```
├── src/                          
│   ├── bin/benchmark_csv.rs      Benchmark driver (CLI entry point)
│   ├── parsers/  
│   │   ├── earley_leo/   
│   │   ├── earley.rs             
│   │   ├── cyk.rs                
│   │   ├── valiant.rs            
│   │   ├── gll/                  
│   │   └── glr/                  
│   ├── grammars.rs               
│   └── parse_tree.rs             
├── grammars/                     Grammar files in JSON format
├── input/                        Benchmark input files (generated corpora)
├── table/                        Pre-generated GLR/LR parse tables
├── results_comprehensive/        Experiment results (CSV files)
├── results/                      Old results (CSV files)
├── plotter.py                   
├── reproduce.sh                  
├── requirements.txt              Python dependencies for plotter
├── rust-toolchain.toml           
├── INSTALL.md                    Installation guide
```

## Parser implementations

| Parser | Algorithm | Key reference |
|---|---|---|
| **Earley** | Chart parsing (Earley 1970) | Adapted from [Gopinath 2021](https://rahul.gopinath.org/post/2021/02/06/earley-parsing/) |
| **GLL** | Graph-structured parsing | ["A Reference GLL Implementation"](https://doi.org/10.1145/3623476.3623521) — Johnstone 2023 |
| **RNGLR** | Right-Nulled GLR | Economopoulos 2006 — [PhD dissertation](https://www.researchgate.net/publication/242287349_Generalised_LR_Parsing_Algorithms) |
| **BRNGLR** | Binarised RNGLR | same |
| **CYK** | Cocke–Younger–Kasami | standard CNF-based algorithm |
| **Valiant** | Matrix multiplication | Valiant 1975, using m4ri for fast multiplication |
| **LL** | LL(1) via GLL framework | standard implementation |
| **LR** | LR(1) via GLR framework | standard implementation |

## Grammars

Multiple grammars are used in the main evaluation.  All are stored in `grammars/` in a JSON representation.  Several were adapted from the [referenceLanguageCorpora](https://github.com/AJohnstone2007/referenceLanguageCorpora) repository.

Additional `ll1_*` and `lr_*` grammar variants are used in the LL/LR baseline experiments.

## Benchmark methodology

The benchmark driver (`src/bin/benchmark_csv.rs`) runs each parser on increasing-length input slices drawn from the corpus files.  For each (parser, input-length) pair it:

1. Performs one warmup iteration (excluded from timing).
2. Runs between 10 and 20 timed iterations, stopping early if 500 ms of wall time have been accumulated.
3. Reports median time (ns) and median absolute deviation (MAD) over the timed iterations.
4. Samples peak memory via a 1 ms polling thread.

Results are written to `results_comprehensive/benchmark_<config>.csv`. Experiment data is included in `results_comprehensive/` and `results/` for reference.

To change the input corpora, grammar, or parsers used, edit the `BenchmarkConfig` struct in `src/bin/benchmark_csv.rs`.

## License

MIT — see [LICENSE](LICENSE).