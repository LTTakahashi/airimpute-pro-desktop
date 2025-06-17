#!/usr/bin/env bash
# Fix Docker network issues in WSL2

set -euo pipefail

echo "Fixing Docker network bridge issues..."

# Stop Docker if running
echo "Stopping Docker service..."
sudo service docker stop 2>/dev/null || true

# Clean up Docker networks
echo "Cleaning up Docker networks..."
sudo rm -rf /var/lib/docker/network/files/*

# Ensure iptables is using legacy mode
echo "Setting iptables to legacy mode..."
sudo update-alternatives --set iptables /usr/sbin/iptables-legacy
sudo update-alternatives --set ip6tables /usr/sbin/ip6tables-legacy

# Start Docker daemon with debug info
echo "Starting Docker daemon..."
sudo dockerd --debug &
DOCKER_PID=$!

# Wait for Docker to be ready
echo "Waiting for Docker to be ready..."
for i in {1..30}; do
    if sudo docker info >/dev/null 2>&1; then
        echo "✓ Docker is ready!"
        break
    fi
    echo -n "."
    sleep 1
done

# Create default bridge network if missing
echo "Ensuring default bridge network exists..."
if ! sudo docker network ls | grep -q bridge; then
    sudo docker network create bridge || true
fi

# Test Docker
echo "Testing Docker..."
if sudo docker run --rm hello-world >/dev/null 2>&1; then
    echo "✓ Docker is working correctly!"
else
    echo "✗ Docker test failed"
    exit 1
fi

echo
echo "Docker network fixed! You can now run:"
echo "  ./build-windows-docker-sudo.sh"