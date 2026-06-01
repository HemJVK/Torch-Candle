# syntax=docker/dockerfile:1.4
# Multi-stage build targeted at PyTorch Parity for Torch-Candle

# Stage 1: Build Environment
FROM nvidia/cuda:12.1.1-devel-ubuntu22.04 AS builder

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV MALLOC_MMAP_THRESHOLD_=65536

# Install core build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    git \
    pkg-config \
    libssl-dev \
    python3 \
    python3-pip \
    python3-dev \
    python3-venv \
    && rm -rf /var/lib/apt/lists/*

# Install Rust toolchain
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

# Install maturin for PyO3 Rust extension compiling
RUN pip3 install --no-cache-dir maturin auditwheel patchelf

WORKDIR /workspace

# Copy workspace files
COPY . .

# Setup virtual environment and compile Torch-Candle
RUN python3 -m venv .venv && \
    .venv/bin/pip install --upgrade pip && \
    .venv/bin/pip install setuptools numpy pytest && \
    USE_CUDA=1 .venv/bin/python3 setup.py install

# Stage 2: Runtime Image
FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04 AS runner

ENV DEBIAN_FRONTEND=noninteractive
ENV MALLOC_MMAP_THRESHOLD_=65536

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-venv \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy python virtual environment and compiled build from builder
COPY --from=builder /workspace/.venv /app/.venv
COPY --from=builder /workspace/src /app/src

# Set environment variables
ENV PATH="/app/.venv/bin:${PATH}"
ENV PYTHONPATH="/app/src:${PYTHONPATH}"

# Configure default command to launch a pytest validation run
CMD ["pytest"]
