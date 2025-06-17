#!/usr/bin/env bash
# Fix Docker on WSL2
# Handles common Docker daemon issues on Windows Subsystem for Linux

set -euo pipefail

# Colors
readonly GREEN='\033[0;32m'
readonly RED='\033[0;31m'
readonly YELLOW='\033[1;33m'
readonly NC='\033[0m'

log() {
    echo -e "${GREEN}[Docker Fix]${NC} $*"
}

error() {
    echo -e "${RED}[ERROR]${NC} $*" >&2
    exit 1
}

warning() {
    echo -e "${YELLOW}[WARNING]${NC} $*"
}

# Check if running on WSL
check_wsl() {
    if grep -qi microsoft /proc/version; then
        log "Detected WSL environment"
        return 0
    else
        warning "This script is designed for WSL. Proceed anyway?"
        read -p "Continue? [y/N] " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 0
        fi
    fi
}

# Start Docker daemon
start_docker_daemon() {
    log "Starting Docker daemon..."
    
    # Check if systemd is available (WSL2 with systemd support)
    if command -v systemctl &> /dev/null && systemctl --version &> /dev/null; then
        log "Using systemd to start Docker..."
        sudo systemctl start docker
        sudo systemctl enable docker
        
        # Check status
        if sudo systemctl is-active --quiet docker; then
            log "✓ Docker daemon started with systemd"
            return 0
        fi
    fi
    
    # Fallback: Start Docker manually (for WSL2 without systemd)
    log "Starting Docker daemon manually..."
    
    # Check if dockerd is already running
    if pgrep -x dockerd > /dev/null; then
        log "Docker daemon is already running"
        return 0
    fi
    
    # Start dockerd in background
    sudo dockerd > /dev/null 2>&1 &
    local dockerd_pid=$!
    
    # Wait for Docker to start
    log "Waiting for Docker daemon to start..."
    local count=0
    while [ $count -lt 30 ]; do
        if sudo docker info > /dev/null 2>&1; then
            log "✓ Docker daemon started successfully"
            return 0
        fi
        sleep 1
        count=$((count + 1))
    done
    
    error "Docker daemon failed to start"
}

# Configure Docker for WSL
configure_wsl_docker() {
    log "Configuring Docker for WSL..."
    
    # Ensure docker group exists
    if ! getent group docker > /dev/null 2>&1; then
        sudo groupadd docker
    fi
    
    # Add user to docker group
    if ! groups | grep -q docker; then
        sudo usermod -aG docker "$USER"
        warning "Added $USER to docker group. You may need to run: newgrp docker"
    fi
    
    # Create Docker daemon config optimized for WSL
    sudo tee /etc/docker/daemon.json > /dev/null <<EOF
{
    "hosts": ["unix:///var/run/docker.sock"],
    "log-driver": "json-file",
    "log-opts": {
        "max-size": "100m",
        "max-file": "3"
    },
    "storage-driver": "overlay2",
    "features": {
        "buildkit": true
    }
}
EOF
}

# Test Docker
test_docker() {
    log "Testing Docker..."
    
    # Test with sudo first
    if sudo docker run --rm hello-world > /dev/null 2>&1; then
        log "✓ Docker is working with sudo"
    else
        error "Docker test failed"
    fi
    
    # Test without sudo (may require newgrp)
    if docker run --rm hello-world > /dev/null 2>&1; then
        log "✓ Docker is working without sudo"
    else
        warning "Docker requires sudo or 'newgrp docker'"
    fi
}

# Create WSL startup script
create_startup_script() {
    log "Creating Docker startup script..."
    
    cat > ~/.docker-wsl-start.sh << 'EOF'
#!/bin/bash
# Start Docker daemon on WSL startup

if ! pgrep -x dockerd > /dev/null; then
    echo "Starting Docker daemon..."
    sudo dockerd > /dev/null 2>&1 &
fi
EOF
    
    chmod +x ~/.docker-wsl-start.sh
    
    # Add to bashrc
    if ! grep -q "docker-wsl-start" ~/.bashrc; then
        echo "" >> ~/.bashrc
        echo "# Auto-start Docker on WSL" >> ~/.bashrc
        echo "[ -f ~/.docker-wsl-start.sh ] && ~/.docker-wsl-start.sh" >> ~/.bashrc
    fi
    
    log "Created startup script: ~/.docker-wsl-start.sh"
}

# Main fix process
main() {
    log "=== Docker WSL Fix ==="
    
    check_wsl
    configure_wsl_docker
    start_docker_daemon
    test_docker
    create_startup_script
    
    echo
    log "=== Docker is now running! ==="
    echo
    echo "Next steps:"
    echo "1. Run: newgrp docker"
    echo "2. Then: ./build-windows-docker.sh"
    echo
    echo "Note: Docker will need to be started manually after WSL restarts."
    echo "Run: ~/.docker-wsl-start.sh"
    echo
}

# Run main function
main "$@"