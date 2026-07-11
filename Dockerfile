# Dockerfile — self-contained artifact for the parser comparison paper.
#
# Build:
#   git submodule update --init --recursive
#   docker build -t parser-comparison .
#
# Open an interactive shell in the prepared environment:
#   docker run --rm -it parser-comparison
#
# Skip benchmarks and only regenerate plots from pre-computed results:
#   docker run --rm parser-comparison bash reproduce.sh --skip-benchmark
#
# Copy results out of the container:
#   docker run --rm -v "$(pwd)/output:/output" parser-comparison \
#       bash -c "bash reproduce.sh && cp -r results plot /output/"
#
# Interactive exploration:
#   docker run --rm -it parser-comparison bash

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
