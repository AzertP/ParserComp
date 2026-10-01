# Dockerfile — containerized environment for building and benchmarking the
# generalized parser implementations in this repository.
#
# Build:
#   git submodule update --init --recursive
#   docker build -t parser-comparison .
#
# Open an interactive shell in the prepared environment:
#   docker run --rm -it parser-comparison
#
# Run the configured valid-input benchmarks and keep the raw CSV results:
#   docker run --rm -v "$(pwd)/results:/artifact/results" parser-comparison \
#       cargo run --release --bin benchmark_csv

FROM rust:1.85.0-bookworm

# Analysis uses only the Python standard library; Valiant uses pure Rust.
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /artifact

# Copy everything
COPY . .

# Build the Rust project in release mode (so reviewers don't need to wait)
RUN cargo build --release --locked

# Default to a shell; users choose which benchmark or analysis command to run.
CMD ["bash"]
