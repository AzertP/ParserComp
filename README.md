# Generalised Parser Comparison

Contains implementations of five generalised parsing algorithm families (CYK,
Valiant, Earley, GLL, and GLR) benchmarked against various context-free
grammars, together with experiment results and input corpora.

DOI: [10.5281/zenodo.19231343](https://doi.org/10.5281/zenodo.19231343)

## Running the benchmarks

After cloning the repository, initialise its Tree-sitter Java submodule:

```bash
git submodule update --init --recursive
```

Alternatively, clone the repository with `git clone --recurse-submodules`.

Install the prerequisites described in [INSTALL.md](INSTALL.md), then build the
release binaries:

```bash
cargo build --release
```

### Select benchmark configurations

The main valid-input driver is configured by the `CONFIGS` array in
`src/bin/benchmark_csv.rs`. Uncomment the `GrammarConfig` entries that you want
to run, and set each entry's `parsers` field to `ALL_PARSERS`, `FAST_PARSERS`,
or a parser-name slice such as `&["Leo", "GLL"]`.

Only enabled entries are run. Each enabled entry writes one CSV file, replacing
an existing file with the same name, so copy any raw result that you need to
preserve before rerunning it.

### Run valid-input benchmarks

```bash
cargo run --release --bin benchmark_csv
```

Results are written to
`results/benchmark_csv/benchmark_<configuration-name>.csv`.

### Run invalid-input benchmarks

Invalid-input benchmarks are configured separately by the `CONFIGS` array in
`src/bin/benchmark_csv_invalid.rs`. Enable the configurations that correspond
to the valid-input runs, then run:

```bash
cargo run --release --bin benchmark_csv_invalid
```

Results are written to
`results/benchmark_csv_invalid/benchmark_<configuration-name>_invalid.csv`.

The other benchmark drivers and their output locations are listed under
[Benchmark Rust files](#benchmark-rust-files).

### Run Tree-sitter comparisons

To run the included `tree-sitter-java` submodule as a Java baseline, run:

```bash
cargo run --release --bin benchmark_tree_sitter_java
```

This writes
`results/benchmark_tree_sitter_java/benchmark_tree_sitter_java.csv` over the same
`input/code/Java/*.java` corpus used by the Java lexical benchmark.

To benchmark only Tree-sitter on the `gamma2` and `gamma3` stress corpora, run:

```bash
cargo run --release --bin benchmark_tree_sitter_stress
```

The command runs both grammars by default. Pass `-- --grammar gamma2` or
`-- --grammar gamma3` to run one grammar. Results are written to
`results/benchmark_tree_sitter_stress/benchmark_tree_sitter_gamma2.csv` and
`results/benchmark_tree_sitter_stress/benchmark_tree_sitter_gamma3.csv`.
Tree-sitter resolves each ambiguous input to one concrete syntax tree; this
benchmark does not construct or enumerate a complete ambiguity forest.

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
├── results/                      Raw benchmark outputs, grouped by driver
├── results_comprehensive/        Combined valid and invalid results
├── results_legacy/               Older experiment results
├── script/                       Result-processing and paper-artifact scripts
├── requirements.txt              Python analysis dependencies
├── rust-toolchain.toml           
├── INSTALL.md                    Installation guide
```

## Parser implementations

| Parser | Algorithm | Key reference |
|---|---|---|
| **Earley / Earley-Leo** | Chart parsing (Earley 1970), with optional Leo right-recursion optimisation | Adapted from [Gopinath 2021](https://rahul.gopinath.org/post/2021/02/06/earley-parsing/) |
| **GLL** | Graph-structured parsing | ["A Reference GLL Implementation"](https://doi.org/10.1145/3623476.3623521) — Johnstone 2023 |
| **RNGLR** | Right-Nulled GLR | Economopoulos 2006 — [PhD dissertation](https://www.researchgate.net/publication/242287349_Generalised_LR_Parsing_Algorithms) |
| **BRNGLR** | Binarised RNGLR | same |
| **CYK** | Cocke–Younger–Kasami | standard CNF-based algorithm |
| **Valiant** | Matrix multiplication | Valiant 1975, using m4ri for fast multiplication |
| **LL** | LL(1) via GLL framework | standard implementation |
| **LR** | LR(1) via GLR framework | standard implementation |

Parser names in benchmark configurations and CSV files are more specific than
the family names above. `Leo` denotes the Earley implementation with Leo's
right-recursion optimisation.
`RNGLR` and `BRNGLR` are the two evaluated implementations in the GLR
family. `LL` and `LR` are deterministic baselines and are not counted among the
five generalised parsing families.

## Grammars

Multiple grammars are used in the main evaluation.  All are stored in `grammars/` in a JSON representation.  Several were adapted from the [referenceLanguageCorpora](https://github.com/AJohnstone2007/referenceLanguageCorpora) repository.

Additional `ll1_*` and `lr_*` grammar variants are used in the LL/LR baseline experiments.

## Benchmark methodology

The benchmark driver (`src/bin/benchmark_csv.rs`) runs each parser on increasing-length input slices drawn from the corpus files.  For each (parser, input-length) pair it:

1. Performs one warmup iteration (excluded from timing).
2. Runs between 10 and 20 timed iterations, stopping early if 500 ms of wall time have been accumulated.
3. Reports median time (ns) and median absolute deviation (MAD) over the timed iterations.
4. Samples peak memory via a 1 ms polling thread.

Raw results are written beneath `results/` in a directory specific to each
benchmark driver. For example, the main driver writes
`results/benchmark_csv/benchmark_<config>.csv`. The repository also includes
combined experiment data in `results_comprehensive/`; see
[Results and paper artifacts](#results-and-paper-artifacts) for the conversion
step.

To change the input corpora, grammar, or parsers used, edit the `CONFIGS` array
of `GrammarConfig` values in `src/bin/benchmark_csv.rs`.

### Benchmark Rust files

| File | Purpose | Output |
|---|---|---|
| `src/bin/benchmark_csv.rs` | Main scannerless benchmarks over generated grammar corpora. | `results/benchmark_csv/benchmark_<grammar>.csv` |
| `src/bin/benchmark_csv_invalid.rs` | Scannerless rejection benchmarks for generated invalid inputs. | `results/benchmark_csv_invalid/benchmark_<grammar>_invalid.csv` |
| `src/bin/benchmark_code.rs` | Whole-file whitespace-aware source-code benchmarks. | `results/benchmark_code/benchmark_<language>.csv` |
| `src/bin/benchmark_lex.rs` | Lexer-first source-code benchmarks using tokenized grammars. | `results/benchmark_lex/benchmark_<language>.csv` |
| `src/bin/benchmark_rlc.rs` | Benchmarks over referenceLanguageCorpora inputs. | `results/benchmark_rlc/benchmark_rlc_<language>.csv` |
| `src/bin/benchmark_tree_sitter_java.rs` | Java comparison against tree-sitter-java. | `results/benchmark_tree_sitter_java/benchmark_tree_sitter_java.csv` |
| `src/bin/benchmark_tree_sitter_stress.rs` | Tree-sitter-only gamma2/gamma3 stress benchmark. | `results/benchmark_tree_sitter_stress/benchmark_tree_sitter_<grammar>.csv` |

### Validation Rust files

| File | Purpose | Output |
|---|---|---|
| `src/bin/validate.rs` | Cross-checks parsers on small test grammars and exhaustive small-language cases. | Console summary; exits nonzero on exhaustive-validation failure. |
| `src/bin/validate_java.rs` | Compares tree-sitter-java with local generalized parsers on extracted OpenJDK parser tests. | `test/openjdk-test/validate_java_result.tsv` |

## Results and paper artifacts

The valid- and invalid-input benchmark drivers write raw CSV files to separate
directories. After running matching configurations in both drivers, combine
their results with:

```bash
python3 script/combine_benchmark_csv.py
```


A pre-rendered overview plot is available here: [allGrammarsGeneralTime.pdf](allGrammarsGeneralTime.pdf)

## Modify the test suite

### Adding new input strings

Each corpus file in `input/` (for example, `input/json.txt`) contains one input
string per line. To test a new input, append it to the relevant file or create a
new file:

```bash
echo '(1+2)*(3+4)' >> input/expr.txt
```

### Adding a new grammar

Grammars are stored as JSON files in `grammars/`. Each file has the following structure (see `grammars/calc.json` for a complete example):

```json
{
  "name": "my_grammar",
  "start": "<start>",
  "rules": {
    "<start>": [["<expr>"]],
    "<expr>":  [["a"], ["(", "<expr>", ")"]]
  }
}
```

Non-terminals are written as `<name>` and terminals are bare strings where each character is treated as a separate token (`"ab"` should be `["a", "b"]`). Some parsers require pre-generated parse tables, which can be generated by setting `generate_table: true` in the benchmark config (see below).

### Running a custom benchmark

Edit the `CONFIGS` array in `src/bin/benchmark_csv.rs` to add a new `GrammarConfig` entry:

```rust
GrammarConfig {
    name: "my_grammar",
    grammar_path: "grammars/my_grammar.json",
    input_paths: &["input/my_grammar.txt"],
    table_path: "table/my_grammar_glr_table.csv",
    lr_table_path: "table/my_grammar_lr_table.csv",
    generate_table: true,
    parsers: ALL_PARSERS,   // or e.g. &["Leo", "GLL"]
}
```

Then rebuild and run:

```bash
cargo run --release --bin benchmark_csv
```

Results are written to `results/benchmark_csv/benchmark_my_grammar.csv`.

## Docker

A Dockerfile is provided, but
does not select or run benchmarks automatically. Enable the desired `CONFIGS`
entries in the Rust source (`src/bin/benchmark_csv.rs`) before building the image:

```bash
git submodule update --init --recursive
docker build -t parser-comparison .
```

Open a shell in the prepared environment with:

```bash
docker run --rm -it parser-comparison
```

Alternatively, name a benchmark command directly. Mount `results/` to retain
raw CSV files after the container exits:

```bash
docker run --rm \
  -v "$(pwd)/results:/artifact/results" \
  parser-comparison \
  cargo run --release --bin benchmark_csv
```

Replace `benchmark_csv` with any binary listed under
[Benchmark Rust files](#benchmark-rust-files). To combine valid and invalid
results while retaining both the raw and combined files, run:

```bash
docker run --rm \
  -v "$(pwd)/results:/artifact/results" \
  -v "$(pwd)/results_comprehensive:/artifact/results_comprehensive" \
  parser-comparison \
  python3 script/combine_benchmark_csv.py
```

## License

MIT — see [LICENSE](LICENSE).
