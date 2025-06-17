#!/usr/bin/env bash
# Complete Docker Build Environment Setup
# This script installs Docker and prepares the Windows build environment

set -euo pipefail

# Colors
readonly GREEN='\033[0;32m'
readonly RED='\033[0;31m'
readonly YELLOW='\033[1;33m'
readonly NC='\033[0m'

log() {
    echo -e "${GREEN}[Setup]${NC} $*"
}

error() {
    echo -e "${RED}[ERROR]${NC} $*" >&2
    exit 1
}

warning() {
    echo -e "${YELLOW}[WARNING]${NC} $*"
}

# Check if Docker is installed
check_docker() {
    if command -v docker &> /dev/null; then
        log "Docker is already installed"
        docker --version
        return 0
    else
        return 1
    fi
}

# Check if user can run Docker without sudo
check_docker_permissions() {
    if docker info &> /dev/null; then
        log "Docker permissions are correctly configured"
        return 0
    else
        if groups | grep -q docker; then
            warning "You're in the docker group but need to refresh your session"
            warning "Run: newgrp docker"
            return 1
        else
            warning "You need to be added to the docker group"
            return 1
        fi
    fi
}

# Install Docker using our script
install_docker() {
    log "Installing Docker..."
    
    local install_script="./install-docker.sh"
    
    if [[ ! -f "$install_script" ]]; then
        error "Docker install script not found: $install_script"
    fi
    
    # Run the installation
    bash "$install_script"
}

# Build Docker image
build_docker_image() {
    log "Building Windows cross-compilation Docker image..."
    
    if [[ ! -f "Dockerfile.windows-build" ]]; then
        error "Dockerfile.windows-build not found"
    fi
    
    # Build the image
    docker build \
        --file Dockerfile.windows-build \
        --tag airimpute-windows-builder:v1.0.0 \
        --build-arg ZIG_VERSION=0.11.0 \
        --build-arg ZIG_SHA256=2d00e789fec4f71790a6e7bf83ff91d564943c5ee843c5fd966efc474b423047 \
        .
    
    log "✓ Docker image built successfully"
}

# Test the build environment
test_build_environment() {
    log "Testing build environment..."
    
    # Run a simple test command in the container
    docker run --rm \
        airimpute-windows-builder:v1.0.0 \
        bash -c "zig version && rustc --version && node --version && makensis -VERSION"
    
    log "✓ Build environment test passed"
}

# Create a quick test script
create_test_script() {
    log "Creating test script..."
    
    cat > test-docker-build.sh << 'EOF'
#!/usr/bin/env bash
# Quick test of the Docker build environment

set -euo pipefail

echo "Testing Docker build environment..."

# Check if we can run Docker
if ! docker info &> /dev/null; then
    echo "ERROR: Cannot run Docker. Try: newgrp docker"
    exit 1
fi

# Run a test build command
docker run --rm \
    --volume "$PWD:/build:rw" \
    --workdir /build \
    --user builder \
    airimpute-windows-builder:v1.0.0 \
    bash -c "echo '✓ Docker environment is working!' && ls -la"

echo "Test completed successfully!"
EOF
    
    chmod +x test-docker-build.sh
    log "Created test-docker-build.sh"
}

# Main setup process
main() {
    log "=== Docker Build Environment Setup ==="
    echo
    
    local needs_docker_install=false
    local needs_newgrp=false
    
    # Check Docker installation
    if ! check_docker; then
        needs_docker_install=true
    fi
    
    # If Docker is installed, check permissions
    if [[ "$needs_docker_install" == "false" ]]; then
        if ! check_docker_permissions; then
            needs_newgrp=true
        fi
    fi
    
    # Install Docker if needed
    if [[ "$needs_docker_install" == "true" ]]; then
        log "Docker is not installed. Starting installation..."
        install_docker
        needs_newgrp=true
    fi
    
    # Handle group membership
    if [[ "$needs_newgrp" == "true" ]]; then
        echo
        warning "=== ACTION REQUIRED ==="
        warning "You need to refresh your session to use Docker without sudo."
        echo
        echo "Please run ONE of the following:"
        echo "  1. newgrp docker  (to refresh current session)"
        echo "  2. Log out and log back in (to refresh all sessions)"
        echo
        echo "After that, run this script again to continue setup."
        exit 0
    fi
    
    # Build Docker image
    log "Building Docker image for Windows cross-compilation..."
    build_docker_image
    
    # Test the environment
    test_build_environment
    
    # Create test script
    create_test_script
    
    # Success message
    echo
    log "=== Setup Complete! ==="
    echo
    echo "Docker build environment is ready."
    echo
    echo "To build Windows executable:"
    echo "  ./build-windows-docker.sh"
    echo
    echo "To test the Docker environment:"
    echo "  ./test-docker-build.sh"
    echo
    log "Happy building! 🚀"
}

# Run main function
main "$@"