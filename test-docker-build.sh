#!/usr/bin/env bash
# Test script to verify Docker build configuration

set -euo pipefail

# Colors
readonly GREEN='\033[0;32m'
readonly RED='\033[0;31m'
readonly YELLOW='\033[1;33m'
readonly NC='\033[0m'

log() {
    echo -e "${GREEN}[Test]${NC} $*"
}

error() {
    echo -e "${RED}[ERROR]${NC} $*" >&2
    exit 1
}

warning() {
    echo -e "${YELLOW}[WARNING]${NC} $*"
}

# Check if Docker is running
check_docker() {
    if ! sudo docker info &> /dev/null; then
        error "Docker is not running. Please run: ./fix-docker-iptables.sh"
    fi
    log "✓ Docker is running"
}

# Test Dockerfile syntax
test_dockerfile() {
    log "Testing Dockerfile syntax..."
    
    if [[ ! -f "Dockerfile.windows-build-v2" ]]; then
        error "Dockerfile.windows-build-v2 not found"
    fi
    
    # Basic syntax check
    if grep -E "^\s*FROM\s+" Dockerfile.windows-build-v2 > /dev/null; then
        log "✓ Dockerfile has valid FROM statement"
    else
        error "Dockerfile missing FROM statement"
    fi
    
    # Check for user switching
    if grep -E "^\s*USER\s+builder" Dockerfile.windows-build-v2 > /dev/null; then
        log "✓ Dockerfile switches to non-root user"
    else
        warning "Dockerfile might not switch to non-root user"
    fi
    
    # Check for Rust installation
    if grep -E "rustup" Dockerfile.windows-build-v2 > /dev/null; then
        log "✓ Dockerfile installs Rust"
    else
        warning "Dockerfile might not install Rust"
    fi
}

# Test build script
test_build_script() {
    log "Testing build script..."
    
    if [[ ! -f "build-windows-secure-v3.sh" ]]; then
        error "build-windows-secure-v3.sh not found"
    fi
    
    if [[ ! -x "build-windows-secure-v3.sh" ]]; then
        warning "build-windows-secure-v3.sh is not executable"
    fi
    
    # Check for correct Python SHA256
    if grep -q "67077e6fa918e4f4fd60ba169820b00be7c390c497bf9bc9cab2c255ea8e6f3e" build-windows-secure-v3.sh; then
        log "✓ Build script has correct Python SHA256"
    else
        error "Build script has incorrect Python SHA256"
    fi
}

# Check if old images exist
check_existing_images() {
    log "Checking for existing Docker images..."
    
    if sudo docker image inspect airimpute-windows-builder:v1.0.0 &> /dev/null; then
        warning "Old image v1.0.0 exists - will be replaced"
    fi
    
    if sudo docker image inspect airimpute-windows-builder:v2.0.0 &> /dev/null; then
        log "✓ New image v2.0.0 already exists"
    else
        log "New image v2.0.0 needs to be built"
    fi
}

# Dry run Docker build
dry_run_docker_build() {
    log "Performing dry run of Docker build..."
    
    # Create a minimal test Dockerfile to verify Docker is working
    cat > Dockerfile.test << 'EOF'
FROM debian:bookworm-slim
RUN echo "Docker build test successful"
EOF
    
    if sudo docker build -f Dockerfile.test -t test-build:latest . &> /dev/null; then
        log "✓ Docker can build images"
        sudo docker rmi test-build:latest &> /dev/null || true
    else
        error "Docker build test failed"
    fi
    
    rm -f Dockerfile.test
}

# Summary
show_summary() {
    echo
    log "=== Test Summary ==="
    log "All checks passed! Ready to build."
    echo
    log "To build the Windows executable, run:"
    echo "  ./build-windows-docker-sudo.sh"
    echo
    log "This will:"
    log "1. Build a Docker image with Rust installed as non-root user"
    log "2. Run the cross-compilation inside the container"
    log "3. Generate Windows executable and installer"
    echo
    log "Expected output:"
    log "- dist/windows/AirImpute-Pro_1.0.0_x64-setup.exe"
    log "- dist/windows/airimpute-pro.exe"
}

# Main
main() {
    log "=== Docker Build Configuration Test ==="
    echo
    
    check_docker
    test_dockerfile
    test_build_script
    check_existing_images
    dry_run_docker_build
    show_summary
}

# Only run if Docker is available
if command -v docker &> /dev/null; then
    main "$@"
else
    error "Docker is not installed"
fi