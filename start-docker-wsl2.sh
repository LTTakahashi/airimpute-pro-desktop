#!/usr/bin/env bash
# Start Docker in WSL2 with proper networking

set -euo pipefail

echo "=== Starting Docker in WSL2 ==="

# Function to check if running in WSL
is_wsl() {
    if grep -qi microsoft /proc/version; then
        return 0
    else
        return 1
    fi
}

# Check if in WSL
if ! is_wsl; then
    echo "Not running in WSL. Using standard Docker start..."
    sudo service docker start
    exit 0
fi

echo "Detected WSL2 environment"

# Stop any existing Docker daemon
echo "Stopping any existing Docker daemon..."
sudo service docker stop 2>/dev/null || true
sudo killall dockerd 2>/dev/null || true

# Configure WSL2 networking
echo "Configuring WSL2 networking..."

# Set iptables to legacy mode (more compatible with WSL2)
sudo update-alternatives --set iptables /usr/sbin/iptables-legacy 2>/dev/null || true
sudo update-alternatives --set ip6tables /usr/sbin/ip6tables-legacy 2>/dev/null || true

# Clean up any corrupted network files
echo "Cleaning up Docker network files..."
sudo rm -rf /var/lib/docker/network/files/* 2>/dev/null || true

# Ensure Docker directory permissions
sudo mkdir -p /var/lib/docker
sudo chmod 755 /var/lib/docker

# Start Docker with specific WSL2 options
echo "Starting Docker daemon with WSL2 options..."
sudo dockerd \
    --iptables=false \
    --bridge=none \
    --log-level=warn \
    >/tmp/docker.log 2>&1 &

DOCKER_PID=$!
echo "Docker daemon PID: $DOCKER_PID"

# Wait for Docker to be ready
echo -n "Waiting for Docker to be ready"
for i in {1..30}; do
    if sudo docker info >/dev/null 2>&1; then
        echo " ✓"
        echo "Docker is ready!"
        break
    fi
    echo -n "."
    sleep 1
done

# Create default network if needed
echo "Setting up Docker networks..."
if ! sudo docker network ls | grep -q bridge; then
    sudo docker network create \
        --driver bridge \
        --subnet=172.17.0.0/16 \
        --gateway=172.17.0.1 \
        bridge 2>/dev/null || true
fi

# Test Docker
echo
echo "Testing Docker..."
if sudo docker run --rm hello-world >/dev/null 2>&1; then
    echo "✓ Docker is working!"
else
    echo "✗ Docker test failed"
    echo "Check logs: sudo tail -f /tmp/docker.log"
    exit 1
fi

# Show network info
echo
echo "Docker networks:"
sudo docker network ls

echo
echo "=== Docker started successfully in WSL2 ==="
echo
echo "You can now run:"
echo "  ./build-windows-docker-sudo.sh"
echo
echo "To check Docker logs:"
echo "  sudo tail -f /tmp/docker.log"