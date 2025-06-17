#!/usr/bin/env bash
# Docker-based Windows Build Script (with sudo)
# This version uses sudo for Docker commands

set -euo pipefail

# Colors
readonly GREEN='\033[0;32m'
readonly RED='\033[0;31m'
readonly YELLOW='\033[1;33m'
readonly NC='\033[0m'

log() {
    echo -e "${GREEN}[Docker Build]${NC} $*"
}

error() {
    echo -e "${RED}[ERROR]${NC} $*" >&2
    exit 1
}

warning() {
    echo -e "${YELLOW}[WARNING]${NC} $*"
}

# Configuration
readonly IMAGE_NAME="airimpute-windows-builder"
readonly IMAGE_TAG="v1.0.0"
readonly CONTAINER_NAME="airimpute-windows-build-$$"

# Get script directory
readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Check Docker
check_docker() {
    if ! sudo docker info &> /dev/null; then
        error "Docker is not running. Please start Docker first."
    fi
}

# Build Docker image
build_docker_image() {
    log "Building Docker image for Windows cross-compilation..."
    
    if [[ ! -f "Dockerfile.windows-build" ]]; then
        error "Dockerfile.windows-build not found"
    fi
    
    # Check if image already exists
    if sudo docker image inspect "${IMAGE_NAME}:${IMAGE_TAG}" &> /dev/null; then
        warning "Docker image already exists. Removing old image..."
        sudo docker rmi "${IMAGE_NAME}:${IMAGE_TAG}" || true
    fi
    
    # Build the image
    if ! sudo docker build \
        --file Dockerfile.windows-build-v2 \
        --tag "${IMAGE_NAME}:${IMAGE_TAG}" \
        --build-arg ZIG_VERSION=0.11.0 \
        --build-arg ZIG_SHA256=2d00e789fec4f71790a6e7bf83ff91d564943c5ee843c5fd966efc474b423047 \
        --build-arg RUST_VERSION=1.82.0 \
        .; then
        error "Docker image build failed"
    fi
    
    log "✓ Docker image built successfully"
}

# Run the build
run_build() {
    log "Running Windows cross-compilation build..."
    
    # Create output directory
    local output_dir="$SCRIPT_DIR/dist/windows"
    mkdir -p "$output_dir"
    
    # Create cache directory
    local cache_dir="${AIRIMPUTE_CACHE_DIR:-$HOME/.cache/airimpute-build}"
    mkdir -p "$cache_dir"
    
    # Ensure cache directory is writable by container user (UID 1000)
    sudo chmod 777 "$cache_dir"
    
    # Run the build
    log "Starting build container..."
    if ! sudo docker run \
        --rm \
        --name "$CONTAINER_NAME" \
        --volume "$SCRIPT_DIR:/build:rw" \
        --volume "$cache_dir:/home/builder/.cache:rw" \
        --workdir /build \
        --user "builder" \
        "${IMAGE_NAME}:${IMAGE_TAG}" \
        ./build-windows-secure-v3.sh; then
        error "Build failed"
    fi
    
    log "✓ Build completed successfully"
}

# Show build output
show_output() {
    local output_dir="$SCRIPT_DIR/dist/windows"
    
    if [[ -d "$output_dir" ]]; then
        log "Build output:"
        ls -la "$output_dir"
        
        # Check for installer
        if [[ -f "$output_dir/AirImpute-Pro_1.0.0_x64-setup.exe" ]]; then
            log "✓ Windows installer created: $output_dir/AirImpute-Pro_1.0.0_x64-setup.exe"
        fi
        
        # Check for executable
        if [[ -f "$output_dir/airimpute-pro.exe" ]]; then
            log "✓ Windows executable created: $output_dir/airimpute-pro.exe"
        fi
    else
        warning "No output directory found"
    fi
}

# Main
main() {
    log "=== AirImpute Pro Docker-based Windows Build (with sudo) ==="
    log "Starting at $(date)"
    echo
    
    check_docker
    build_docker_image
    run_build
    show_output
    
    echo
    log "=== Build Complete ==="
    log "Finished at $(date)"
    echo
    log "Note: Using sudo for Docker commands."
    log "To avoid sudo, run: newgrp docker (requires your user password)"
}

# Run main
main "$@"