#!/usr/bin/env bash
# Docker Installation Script for Ubuntu
# Installs Docker Engine and configures it for the current user

set -euo pipefail

# Colors for output
readonly GREEN='\033[0;32m'
readonly RED='\033[0;31m'
readonly YELLOW='\033[1;33m'
readonly NC='\033[0m'

log() {
    echo -e "${GREEN}[Docker Install]${NC} $*"
}

error() {
    echo -e "${RED}[ERROR]${NC} $*" >&2
    exit 1
}

warning() {
    echo -e "${YELLOW}[WARNING]${NC} $*"
}

# Check if running as root
check_root() {
    if [[ $EUID -eq 0 ]]; then
        error "This script should not be run as root. Please run as a regular user."
    fi
}

# Remove old Docker versions
remove_old_docker() {
    log "Removing old Docker versions if present..."
    
    local old_packages=(
        "docker"
        "docker-engine"
        "docker.io"
        "containerd"
        "runc"
    )
    
    for pkg in "${old_packages[@]}"; do
        if dpkg -l | grep -q "^ii.*$pkg"; then
            log "Removing $pkg..."
            sudo apt-get remove -y "$pkg" || true
        fi
    done
}

# Install prerequisites
install_prerequisites() {
    log "Installing prerequisites..."
    
    sudo apt-get update
    sudo apt-get install -y \
        ca-certificates \
        curl \
        gnupg \
        lsb-release
}

# Add Docker's official GPG key
add_docker_gpg_key() {
    log "Adding Docker's official GPG key..."
    
    # Create keyrings directory
    sudo install -m 0755 -d /etc/apt/keyrings
    
    # Download and add GPG key
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg | \
        sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
    
    sudo chmod a+r /etc/apt/keyrings/docker.gpg
}

# Add Docker repository
add_docker_repository() {
    log "Adding Docker repository..."
    
    local arch
    arch=$(dpkg --print-architecture)
    
    echo \
        "deb [arch=$arch signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
        $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
        sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
}

# Install Docker Engine
install_docker() {
    log "Installing Docker Engine..."
    
    sudo apt-get update
    
    # Install Docker packages
    sudo apt-get install -y \
        docker-ce \
        docker-ce-cli \
        containerd.io \
        docker-buildx-plugin \
        docker-compose-plugin
    
    # Verify installation
    if ! command -v docker &> /dev/null; then
        error "Docker installation failed"
    fi
    
    log "Docker installed successfully"
    sudo docker version
}

# Configure Docker for non-root user
configure_user_access() {
    log "Configuring Docker for user: $USER"
    
    # Create docker group if it doesn't exist
    if ! getent group docker > /dev/null 2>&1; then
        sudo groupadd docker
    fi
    
    # Add current user to docker group
    sudo usermod -aG docker "$USER"
    
    log "User $USER added to docker group"
    warning "You need to log out and back in for group changes to take effect"
    warning "Or run: newgrp docker"
}

# Configure Docker daemon
configure_docker_daemon() {
    log "Configuring Docker daemon..."
    
    # Create daemon configuration
    sudo tee /etc/docker/daemon.json > /dev/null <<EOF
{
    "log-driver": "json-file",
    "log-opts": {
        "max-size": "100m",
        "max-file": "3"
    },
    "default-ulimits": {
        "nofile": {
            "Name": "nofile",
            "Hard": 64000,
            "Soft": 64000
        }
    },
    "storage-driver": "overlay2",
    "features": {
        "buildkit": true
    }
}
EOF
    
    # Restart Docker to apply configuration
    sudo systemctl restart docker
    sudo systemctl enable docker
    
    log "Docker daemon configured"
}

# Test Docker installation
test_docker() {
    log "Testing Docker installation..."
    
    # Try to run hello-world container
    if sudo docker run --rm hello-world > /dev/null 2>&1; then
        log "✓ Docker is working correctly"
    else
        error "Docker test failed"
    fi
}

# Post-installation instructions
show_post_install_instructions() {
    echo
    log "=== Docker Installation Complete ==="
    echo
    echo "To use Docker without sudo, you need to:"
    echo "1. Log out and log back in, OR"
    echo "2. Run: newgrp docker"
    echo
    echo "Then test with: docker run hello-world"
    echo
    echo "To build the Windows executable:"
    echo "  ./build-windows-docker.sh"
    echo
}

# Main installation process
main() {
    log "=== Docker Installation for Ubuntu ==="
    log "This will install Docker Engine and configure it for your user"
    echo
    
    # Check prerequisites
    check_root
    
    # Confirm installation
    read -p "Continue with Docker installation? [y/N] " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        log "Installation cancelled"
        exit 0
    fi
    
    # Run installation steps
    remove_old_docker
    install_prerequisites
    add_docker_gpg_key
    add_docker_repository
    install_docker
    configure_user_access
    configure_docker_daemon
    test_docker
    
    # Show completion message
    show_post_install_instructions
}

# Run main function
main "$@"