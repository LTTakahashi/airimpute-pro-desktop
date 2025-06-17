#!/usr/bin/env bash
# Docker build with retry logic for network issues

set -euo pipefail

# Colors
readonly GREEN='\033[0;32m'
readonly RED='\033[0;31m'
readonly YELLOW='\033[1;33m'
readonly NC='\033[0m'

log() {
    echo -e "${GREEN}[Build]${NC} $*"
}

error() {
    echo -e "${RED}[ERROR]${NC} $*" >&2
}

warning() {
    echo -e "${YELLOW}[WARNING]${NC} $*"
}

# Configuration
readonly IMAGE_NAME="airimpute-windows-builder"
readonly IMAGE_TAG="v2.0.0"
readonly MAX_RETRIES=3

# Check Docker
check_docker() {
    log "Checking Docker status..."
    
    if ! sudo docker info &>/dev/null; then
        error "Docker is not running!"
        echo
        echo "Please start Docker first:"
        echo "  sudo ./start-docker-wsl2.sh"
        echo
        echo "Or if that fails:"
        echo "  sudo ./fix-docker-network.sh"
        exit 1
    fi
    
    log "✓ Docker is running"
    
    # Test network
    log "Testing Docker network..."
    if ! sudo docker run --rm alpine ping -c 1 google.com &>/dev/null; then
        warning "Docker network might have issues"
        echo "Trying to fix..."
        sudo docker network prune -f
        sudo docker network create bridge 2>/dev/null || true
    fi
}

# Build with retry
build_with_retry() {
    local retry_count=0
    
    while [ $retry_count -lt $MAX_RETRIES ]; do
        log "Build attempt $((retry_count + 1)) of $MAX_RETRIES..."
        
        if sudo docker build \
            --network=host \
            --file Dockerfile.windows-build-v2 \
            --tag "${IMAGE_NAME}:${IMAGE_TAG}" \
            --build-arg ZIG_VERSION=0.11.0 \
            --build-arg ZIG_SHA256=2d00e789fec4f71790a6e7bf83ff91d564943c5ee843c5fd966efc474b423047 \
            --build-arg RUST_VERSION=1.82.0 \
            .; then
            log "✓ Docker image built successfully!"
            return 0
        else
            retry_count=$((retry_count + 1))
            if [ $retry_count -lt $MAX_RETRIES ]; then
                warning "Build failed, retrying in 5 seconds..."
                sleep 5
                
                # Try to fix common issues
                log "Attempting to fix Docker networking..."
                sudo docker network prune -f
                sudo service docker restart
                sleep 10
            fi
        fi
    done
    
    error "Build failed after $MAX_RETRIES attempts"
    return 1
}

# Run the build
run_build() {
    log "Running Windows cross-compilation..."
    
    # Create directories
    mkdir -p build-windows dist/windows
    local cache_dir="${HOME}/.cache/airimpute-build"
    mkdir -p "$cache_dir"
    sudo chmod 777 "$cache_dir"
    
    # Run build
    if sudo docker run \
        --rm \
        --network=host \
        --name "airimpute-build-$$" \
        --volume "$PWD:/build:rw" \
        --volume "$cache_dir:/cache:rw" \
        --workdir /build \
        --user builder \
        "${IMAGE_NAME}:${IMAGE_TAG}" \
        ./build-windows-secure-v3.sh; then
        log "✓ Build completed successfully!"
        return 0
    else
        error "Build failed"
        return 1
    fi
}

# Main
main() {
    log "=== AirImpute Pro Windows Build (with retry) ==="
    echo
    
    check_docker
    
    # Remove old images
    if sudo docker image inspect "${IMAGE_NAME}:${IMAGE_TAG}" &>/dev/null; then
        warning "Removing old Docker image..."
        sudo docker rmi "${IMAGE_NAME}:${IMAGE_TAG}" || true
    fi
    
    # Build image
    if ! build_with_retry; then
        error "Failed to build Docker image"
        echo
        echo "Troubleshooting:"
        echo "1. Check Docker logs: sudo journalctl -u docker"
        echo "2. Test network: ./test-docker-network.sh"
        echo "3. Restart Docker: sudo ./start-docker-wsl2.sh"
        exit 1
    fi
    
    # Run build
    if ! run_build; then
        error "Failed to run build"
        exit 1
    fi
    
    # Show results
    echo
    log "=== Build Complete ==="
    if [ -f "dist/windows/airimpute-pro.exe" ]; then
        log "✓ Windows executable: dist/windows/airimpute-pro.exe"
    fi
    if [ -f "dist/windows/AirImpute-Pro_1.0.0_x64-setup.exe" ]; then
        log "✓ Windows installer: dist/windows/AirImpute-Pro_1.0.0_x64-setup.exe"
    fi
}

# Run
main "$@"