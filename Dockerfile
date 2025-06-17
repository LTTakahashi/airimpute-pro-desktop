# Multi-stage Dockerfile for AirImpute Pro Desktop
# Implements reproducible builds following scientific software engineering standards

# ============================================
# Stage 1: Base build environment
# ============================================
FROM ubuntu:24.04 AS base

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive
ENV RUST_BACKTRACE=1
ENV CARGO_HOME=/usr/local/cargo
ENV PATH=/usr/local/cargo/bin:$PATH
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    # Build essentials
    build-essential \
    curl \
    wget \
    git \
    pkg-config \
    # Tauri/WebKit dependencies
    libwebkit2gtk-4.0-dev \
    libjavascriptcoregtk-4.0-dev \
    libgtk-3-dev \
    libayatana-appindicator3-dev \
    librsvg2-dev \
    libasound2-dev \
    libssl-dev \
    # Python and scientific computing
    python3.11 \
    python3.11-dev \
    python3-pip \
    python3-numpy \
    python3-pandas \
    python3-scipy \
    # Additional tools
    clang \
    cmake \
    ninja-build \
    && rm -rf /var/lib/apt/lists/*

# Install Rust
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain=1.75.0
RUN rustup component add rustfmt clippy

# Install Node.js
RUN curl -fsSL https://deb.nodesource.com/setup_18.x | bash - \
    && apt-get install -y nodejs \
    && npm install -g npm@latest

# ============================================
# Stage 2: Python scientific environment
# ============================================
FROM base AS python-env

WORKDIR /app

# Create virtual environment for reproducibility
RUN python3.11 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy Python requirements
COPY requirements.txt requirements-dev.txt ./

# Install Python packages with exact versions
RUN pip install --upgrade pip setuptools wheel \
    && pip install -r requirements.txt \
    && pip install -r requirements-dev.txt

# ============================================
# Stage 3: Frontend builder
# ============================================
FROM base AS frontend-builder

WORKDIR /app

# Copy package files
COPY package.json package-lock.json ./

# Install Node dependencies
RUN npm ci --no-audit

# Copy frontend source
COPY tsconfig*.json vite.config.ts tailwind.config.js postcss.config.js ./
COPY index.html ./
COPY public/ ./public/
COPY src/ ./src/

# Build frontend
RUN npm run build:frontend

# ============================================
# Stage 4: Rust builder
# ============================================
FROM base AS rust-builder

WORKDIR /app

# Copy Rust project files
COPY src-tauri/Cargo.toml src-tauri/Cargo.lock ./src-tauri/
COPY src-tauri/tauri.conf.json ./src-tauri/
COPY src-tauri/build.rs ./src-tauri/

# Create dummy main.rs to cache dependencies
RUN mkdir -p src-tauri/src && \
    echo "fn main() {}" > src-tauri/src/main.rs

# Build dependencies
WORKDIR /app/src-tauri
RUN cargo build --release
RUN rm -rf src

# Copy actual source code
COPY src-tauri/src ./src

# Touch main.rs to ensure rebuild
RUN touch src/main.rs

# Build release binary
RUN cargo build --release

# ============================================
# Stage 5: Test runner
# ============================================
FROM python-env AS test-runner

WORKDIR /app

# Copy all project files
COPY . .

# Copy built artifacts from previous stages
COPY --from=frontend-builder /app/dist ./dist
COPY --from=rust-builder /app/src-tauri/target ./src-tauri/target

# Run all tests
RUN npm test -- --coverage
RUN cd src-tauri && cargo test --release
RUN pytest tests/ -v --cov --cov-report=xml

# ============================================
# Stage 6: Security scanner
# ============================================
FROM base AS security-scanner

WORKDIR /app

# Install security tools
RUN pip install bandit safety
RUN npm install -g snyk retire

# Copy project files
COPY . .

# Run security scans
RUN bandit -r scripts/ -f json -o security-report-python.json || true
RUN safety check --json > security-report-deps.json || true
RUN npm audit --json > security-report-npm.json || true
RUN cd src-tauri && cargo audit --json > ../security-report-rust.json || true

# ============================================
# Stage 7: Documentation builder
# ============================================
FROM python-env AS docs-builder

WORKDIR /app

# Install documentation tools
RUN pip install sphinx sphinx-rtd-theme sphinx-autodoc-typehints
RUN npm install -g typedoc

# Copy source files
COPY . .

# Generate documentation
RUN typedoc --out docs/api/frontend src/
RUN cd src-tauri && cargo doc --no-deps
RUN sphinx-build -b html docs/source docs/build/html

# ============================================
# Stage 8: Final packager
# ============================================
FROM base AS packager

WORKDIR /app

# Copy everything needed for packaging
COPY --from=frontend-builder /app/dist ./dist
COPY --from=rust-builder /app/src-tauri/target/release/airimpute-pro ./src-tauri/target/release/
COPY --from=test-runner /app/coverage ./coverage
COPY --from=security-scanner /app/security-report-*.json ./security-reports/
COPY --from=docs-builder /app/docs ./docs

# Copy packaging scripts and resources
COPY icons/ ./icons/
COPY installer/ ./installer/
COPY LICENSE README.md ./

# Create AppImage
RUN if [ "$(uname -m)" = "x86_64" ]; then \
    wget https://github.com/AppImage/AppImageKit/releases/download/continuous/appimagetool-x86_64.AppImage -O /usr/local/bin/appimagetool && \
    chmod +x /usr/local/bin/appimagetool; \
    fi

# Package application
RUN mkdir -p packages
CMD ["bash", "-c", "npm run tauri build -- --target all"]

# ============================================
# Stage 9: Development environment
# ============================================
FROM python-env AS development

WORKDIR /app

# Install development tools
RUN apt-get update && apt-get install -y \
    vim \
    tmux \
    htop \
    gdb \
    valgrind \
    && rm -rf /var/lib/apt/lists/*

# Install Rust development tools
RUN cargo install cargo-watch cargo-expand cargo-criterion

# Copy project files
COPY . .

# Install all dependencies
RUN npm ci
RUN cd src-tauri && cargo fetch

# No webkit compatibility symlinks needed since we're using webkit 4.0 directly

# Set up development environment
ENV RUST_LOG=debug
ENV NODE_ENV=development
ENV RUSTFLAGS="-C link-arg=-lpython3.11"
ENV PYO3_PYTHON=/usr/bin/python3.11

# Expose ports
EXPOSE 5173 1420

# Default command for development
CMD ["npm", "run", "dev"]