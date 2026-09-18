# Installation Guide

## Option A — Docker (recommended)

The Docker image captures the exact environment used for the paper.

```bash
git submodule update --init --recursive
docker build -t parser-comparison .
```

Open an interactive shell:

```bash
docker run --rm -it parser-comparison
```

Run a benchmark and write results to the host:

```bash
docker run --rm \
    -v "$(pwd)/results:/artifact/results" \
    parser-comparison \
    cargo run --release --bin benchmark_csv
```

---

## Option B — Native install

### Prerequisites

| Tool | Version | Notes |
|---|---|---|
| Rust (via rustup) | 1.85 stable | https://rustup.rs |
| Python 3 | any recent 3.x | system package or https://python.org |
| C compiler | platform default | required by the Tree-sitter grammars |

### 1. Rust

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source "$HOME/.cargo/env"
```

[rust-toolchain.toml](rust-toolchain.toml) pins the channel to `stable`, so `rustup` selects the correct toolchain automatically inside this directory.

### 2. Python packages

```bash
pip install -r requirements.txt
```

This installs `pandas`, `matplotlib`, and `numpy`.

### 3. Build

```bash
cargo build --release
```

### 4. Verify

```bash
cargo build --release                  # exits 0
python3 -c "import pandas, matplotlib, numpy; print('OK')"
```
