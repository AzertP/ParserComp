#!/usr/bin/env bash
# reproduce.sh — Full reproduction pipeline for the parser comparison paper.
#
# Usage:
#   bash reproduce.sh [--skip-benchmark]
#
#   --skip-benchmark   Skip the Rust benchmark step and use the pre-computed
#                      CSV files already present in results/.  Use this for a
#                      quick check that the plots regenerate correctly.
#
# Output:
#   results/benchmark_<grammar>_<size>.csv   (one file per benchmark config)
#   plot/*.pdf                               (all paper figures)

set -euo pipefail

SKIP_BENCHMARK=false
for arg in "$@"; do
  [[ "$arg" == "--skip-benchmark" ]] && SKIP_BENCHMARK=true
done

echo "========================================"
echo " Parser Comparison — Reproduction Script"
echo "========================================"

# --------------------------------------------------------------------------
# Step 1: Build
# --------------------------------------------------------------------------
echo ""
echo "[1/3] Building Rust benchmarking tool (release mode)..."
cargo build --release
echo "      Build complete."

# --------------------------------------------------------------------------
# Step 2: Benchmark
# --------------------------------------------------------------------------
if [[ "$SKIP_BENCHMARK" == true ]]; then
  echo ""
  echo "[2/3] Skipping benchmarks — using pre-computed results in results/."
else
  echo ""
  echo "[2/3] Running benchmarks..."
  echo "      This will write CSV files to results/."
  echo ""
  cargo run --release --bin benchmark_csv
  echo "      Benchmarks complete."
fi
