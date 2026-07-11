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

FROM rust:1.85-bookworm

# System dependencies: m4ri (for Valiant parser), Python 3, pkg-config
RUN apt-get update && apt-get install -y --no-install-recommends \
        libm4ri-dev \
        pkg-config \
        python3 \
        python3-pip \
        python3-venv \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /artifact

# Copy everything
COPY . .

# Install Python dependencies
RUN python3 -m pip install --break-system-packages --no-cache-dir -r requirements.txt

# Build the Rust project in release mode (so reviewers don't need to wait)
RUN cargo build --release

# Default to a shell; users choose which benchmark or analysis command to run.
CMD ["bash"]
