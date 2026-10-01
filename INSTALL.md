# Installation

Run commands from the repository root.

## Docker

Requires Docker.

```bash
git submodule update --init --recursive
docker build -t parser-comparison .
docker run --rm parser-comparison cargo test --release --locked --offline
```

Open a shell:

```bash
docker run --rm -it parser-comparison
```

## Native

Requires Rust 1.85.0 via [rustup](https://rustup.rs), a C compiler, and Python
3.10+ for analysis. No third-party Python packages are required.

```bash
git submodule update --init --recursive
cargo build --release --locked
cargo test --release --locked
```

## Quick check

Run natively or inside the container:

```bash
target/release/parser_comparison --text '(1+2)*3' grammars/calc.json --char
```

Expected output: `Parse succeeded`.

See [README.md](README.md) for benchmarks and result generation.
