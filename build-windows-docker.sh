#!/usr/bin/env bash
# Docker-based Windows Cross-Compilation for AirImpute Pro
# This script uses the secure Docker environment to build Windows executables

set -euo pipefail

# Configuration
readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly IMAGE_NAME="airimpute-windows-builder"
readonly IMAGE_TAG="v1.0.0"
readonly CONTAINER_NAME="airimpute-build-${RANDOM}"

# Colors for output
readonly GREEN='\033[0;32m'
readonly RED='\033[0;31m'
readonly NC='\033[0m'

log() {
    echo -e "${GREEN}[Docker Build]${NC} $*"
}

error() {
    echo -e "${RED}[ERROR]${NC} $*" >&2
    exit 1
}

# Build Docker image
build_docker_image() {
    log "Building Docker image for Windows cross-compilation..."
    
    cd "$SCRIPT_DIR"
    
    if [[ ! -f "Dockerfile.windows-build" ]]; then
        error "Dockerfile.windows-build not found"
    fi
    
    docker build \
        --file Dockerfile.windows-build \
        --tag "${IMAGE_NAME}:${IMAGE_TAG}" \
        --build-arg ZIG_VERSION=0.11.0 \
        --build-arg ZIG_SHA256=2d00e789fec4f71790a6e7bf83ff91d564943c5ee843c5fd966efc474b423047 \
        . || error "Docker image build failed"
    
    log "✓ Docker image built successfully"
}

# Run build in Docker container
run_build() {
    log "Running Windows build in Docker container..."
    
    # Prepare volume mounts
    local cache_dir="${HOME}/.cache/airimpute-docker-build"
    mkdir -p "$cache_dir"
    
    # Ensure cache directory is writable by container user (UID 1000)
    chmod 777 "$cache_dir"
    
    # Run the build
    docker run \
        --rm \
        --name "$CONTAINER_NAME" \
        --volume "${SCRIPT_DIR}:/build:rw" \
        --volume "${cache_dir}:/cache:rw" \
        --env CARGO_HOME=/cache/cargo \
        --env RUSTUP_HOME=/cache/rustup \
        --env PIP_CACHE_DIR=/cache/pip \
        --env NPM_CONFIG_CACHE=/cache/npm \
        --env CODE_SIGNING_CERT_FILE="${CODE_SIGNING_CERT_FILE:-}" \
        --env CODE_SIGNING_KEY_FILE="${CODE_SIGNING_KEY_FILE:-}" \
        --env CODE_SIGNING_KEY_PASSWORD="${CODE_SIGNING_KEY_PASSWORD:-}" \
        --env TAURI_PRIVATE_KEY="${TAURI_PRIVATE_KEY:-}" \
        --env TAURI_KEY_PASSWORD="${TAURI_KEY_PASSWORD:-}" \
        --workdir /build \
        --user "builder" \
        "${IMAGE_NAME}:${IMAGE_TAG}" \
        ./build-windows-secure-v3.sh || error "Build failed"
    
    log "✓ Build completed successfully"
}

# Copy artifacts
copy_artifacts() {
    log "Copying build artifacts..."
    
    local output_dir="${SCRIPT_DIR}/dist/windows"
    mkdir -p "$output_dir"
    
    # Copy executables
    if [[ -d "${SCRIPT_DIR}/build-windows/target/x86_64-pc-windows-msvc/release" ]]; then
        find "${SCRIPT_DIR}/build-windows/target/x86_64-pc-windows-msvc/release" \
            -name "*.exe" -o -name "*.msi" | while read -r file; do
            cp -v "$file" "$output_dir/"
        done
    fi
    
    # Copy installer
    local nsis_dir="${SCRIPT_DIR}/build-windows/target/x86_64-pc-windows-msvc/release/bundle/nsis"
    if [[ -d "$nsis_dir" ]]; then
        local installer
        installer=$(find "$nsis_dir" -name "*.exe" -print -quit)
        if [[ -n "$installer" ]]; then
            cp -v "$installer" "$output_dir/"
        else
            error "NSIS installer .exe not found in $nsis_dir"
        fi
    else
        warning "NSIS bundle directory not found. Skipping installer copy."
    fi
    
    log "✓ Artifacts copied to: $output_dir"
    ls -la "$output_dir"
}

# Main function
main() {
    log "=== AirImpute Pro Docker-based Windows Build ==="
    log "Starting at $(date)"
    
    # Check Docker availability
    if ! command -v docker >/dev/null 2>&1; then
        error "Docker is not installed or not in PATH"
    fi
    
    # Build or update Docker image
    build_docker_image
    
    # Run the build
    run_build
    
    # Copy artifacts
    copy_artifacts
    
    log "=== Build completed successfully ==="
    log "Windows installers are in: dist/windows/"
}

# Run main function
main "$@"