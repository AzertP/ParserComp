# Installation Guide

## Prerequisites

| Tool | Minimum version | Install |
|---|---|---|
| Rust (via rustup) | 1.85 (stable) | https://rustup.rs |
| Python | 3.11 | https://python.org |
| m4ri C library | any recent | see below |

### 1. Rust

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source "$HOME/.cargo/env"
```

The [rust-toolchain.toml](rust-toolchain.toml) file pins the channel to `stable`, so `rustup` will select the correct toolchain automatically when you run any `cargo` command inside this directory.

### 2. Python packages

```bash
pip install -r requirements.txt
```

### 3. m4ri (required by the Valiant parser)

The `m4ri-rust` crate wraps the [m4ri](https://malb.bitbucket.io/m4ri/) C library for fast matrix operations over GF(2).

**macOS (Homebrew):**
```bash
brew install m4ri
```

**Ubuntu / Debian:**
```bash
sudo apt-get install libm4ri-dev
```

**Other Linux:**  
Build from source: https://github.com/malb/m4ri

After installing m4ri, verify that `pkg-config --modversion m4ri` prints a version number.

### 4. Verify the setup

```bash
cargo build --release          # should complete without errors
python3 -c "import pandas, matplotlib, numpy; print('OK')"
```
