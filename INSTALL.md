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
| pkg-config | any | required to locate m4ri at build time |
| m4ri C library | any recent | see below |

### 1. Rust

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source "$HOME/.cargo/env"
```

[rust-toolchain.toml](rust-toolchain.toml) pins the channel to `stable`, so `rustup` selects the correct toolchain automatically inside this directory.

### 2. m4ri and pkg-config (required by the Valiant parser)

The `m4ri-rust` crate wraps the [m4ri](https://malb.bitbucket.io/m4ri/) C library for fast matrix operations over GF(2).

**Ubuntu / Debian:**
```bash
sudo apt-get install libm4ri-dev pkg-config
```

**macOS (Homebrew, if available):**
```bash
brew install m4ri pkg-config
```

If that fails (`No available formula with the name "m4ri"`), build from source:
```bash
brew install autoconf automake libtool pkg-config
git clone https://github.com/malb/m4ri
cd m4ri && autoreconf --install && ./configure && make && sudo make install
cd ..
```

**Other Linux:**  
Build from source: https://github.com/malb/m4ri — then ensure `pkg-config --modversion m4ri` prints a version number.

### 3. Python packages

```bash
pip install -r requirements.txt
```

This installs `pandas`, `matplotlib`, and `numpy`.

### 4. Build

```bash
cargo build --release
```

### 5. Verify

```bash
pkg-config --modversion m4ri          # prints a version number
cargo build --release                  # exits 0
python3 -c "import pandas, matplotlib, numpy; print('OK')"
```
