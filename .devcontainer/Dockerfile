FROM ubuntu:22.04

# Avoid interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install base dependencies, Python, and the correct WebKit version
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    wget \
    file \
    git \
    libssl-dev \
    libgtk-3-dev \
    # This is the key part: specify the version needed by the build
    libwebkit2gtk-4.0-dev \
    libsoup2.4-dev \
    libjavascriptcoregtk-4.0-dev \
    librsvg2-dev \
    patchelf \
    python3 \
    python3-pip \
    python3-venv \
    pkg-config \
    libdbus-1-dev \
    libappindicator3-dev \
    libayatana-appindicator3-dev

# Install Rust
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

# Set up a non-root user (good practice)
RUN useradd -ms /bin/bash vscode
USER vscode
WORKDIR /home/vscode/project

# Install Rust for the vscode user
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/home/vscode/.cargo/bin:${PATH}"

# Your application setup can go here
# e.g., COPY pyproject.toml .
# RUN pip3 install --user -r requirements.txt