# Replication Package — Generalised Parser Comparison

Replication package for the ASE paper submission. Contains the full implementation of five generalised parsing algorithms (CYK, Valiant, Earley, GLL, RNGLR, BRNGLR) benchmarked against eight context-free grammars, together with the pre-computed results and plotting scripts needed to reproduce all paper figures.

## Quick start (pre-computed results → figures)

```bash
# 1. Install prerequisites (see INSTALL.md)
pip install -r requirements.txt

# 2. Regenerate all plots from the included CSV data
python3 plotter.py
# Figures are written to plot/
```

## Full reproduction (build → benchmark → plot)

```bash
bash reproduce.sh
```

Or step by step:

```bash
cargo build --release
cargo run --release --bin benchmark_csv   # writes results/*.csv
python3 plotter.py                         # writes plot/*.pdf
```

See [INSTALL.md](INSTALL.md) for detailed prerequisite setup (Rust, Python, m4ri).  
See [CLAIMS.md](CLAIMS.md) for a figure-to-script/data mapping.

## Repository structure

```
├── src/                          Rust source
│   ├── bin/benchmark_csv.rs      Benchmark driver (CLI entry point)
│   ├── parsers/                  Six parser implementations
│   │   ├── earley.rs             Earley
│   │   ├── cyk.rs                CYK
│   │   ├── valiant.rs            Valiant
│   │   ├── gll/                  GLL + LL
│   │   └── glr/                  RNGLR, BRNGLR + LR
│   ├── grammars.rs               Grammar loader (JSON → internal repr.)
│   └── parse_tree.rs             Shared parse-tree type
├── grammars/                     Grammar files in JSON format
├── input/                        Benchmark input files (generated corpora)
├── table/                        Pre-generated GLR/LR parse tables
├── results/                      Pre-computed benchmark CSVs (included)
├── plotter.py                    Generates all paper figures from results/
├── reproduce.sh                  End-to-end reproduction script
├── requirements.txt              Python dependencies
├── rust-toolchain.toml           Pins Rust stable toolchain
├── INSTALL.md                    Prerequisite installation guide
└── CLAIMS.md                     Maps paper figures to data and scripts
```

## Parser implementations

| Parser | Algorithm | Key reference |
|---|---|---|
| **Earley** | Chart parsing (Earley 1970) | Adapted from [Gopinath 2021](https://rahul.gopinath.org/post/2021/02/06/earley-parsing/) |
| **GLL** | Graph-structured parsing | ["A Reference GLL Implementation"](https://doi.org/10.1145/3623476.3623521) — Johnstone 2023 |
| **RNGLR** | Right-Nulled GLR | Economopoulos 2006 — [PhD dissertation](https://www.researchgate.net/publication/242287349_Generalised_LR_Parsing_Algorithms) |
| **BRNGLR** | Binarised RNGLR | same |
| **CYK** | Cocke–Younger–Kasami | standard CNF-based algorithm |
| **Valiant** | Matrix multiplication | Valiant 1975, using m4ri for GF(2) multiplication |
| **LL** | LL(1) via GLL framework | driven by LL(1) parse table |
| **LR** | LR(1) via GLR framework | driven by LR(1) parse table |

## Grammars

Eight grammars are used in the main evaluation.  All are stored in `grammars/` in a JSON representation.  Several were adapted from the [referenceLanguageCorpora](https://github.com/AJohnstone2007/referenceLanguageCorpora) repository.

| Grammar key | Language | File |
|---|---|---|
| `sexp` | S-Expressions | `grammars/sexp.json` |
| `calc` | Calculator expressions | `grammars/calc.json` |
| `tinyc` | TinyC | `grammars/tinyc.json` |
| `json` | JSON | `grammars/json.json` |
| `ansi_c` | ANSI C | `grammars/ansi_c.json` |
| `cpp` | C++ | `grammars/cpp.json` |
| `java` | Java (JLS 18) | `grammars/jsl18.json` |
| `pascal` | Pascal | `grammars/pascal.json` |

Additional `ll1_*` and `lr_*` grammar variants are used in the LL/LR baseline experiments.

## Benchmark methodology

The benchmark driver (`src/bin/benchmark_csv.rs`) runs each parser on increasing-length input slices drawn from the corpus files.  For each (parser, input-length) pair it:

1. Performs one warmup iteration (excluded from timing).
2. Runs between 5 and 20 timed iterations, stopping early if 500 ms of wall time have been accumulated.
3. Reports **median time** (ns) and **MAD** over the timed iterations.
4. Samples **peak memory** via a 1 ms polling thread.

Results are written to `results/benchmark_<config>.csv`.  Pre-computed CSVs for all configurations are included in the repository so reviewers can reproduce the figures without re-running the full benchmark (which takes 10–60 minutes depending on hardware).

## License

MIT — see [LICENSE](LICENSE).