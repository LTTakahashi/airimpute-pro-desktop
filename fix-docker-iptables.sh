#!/usr/bin/env bash
# Fix Docker iptables issues in WSL2
# This script resolves common iptables problems that prevent Docker from starting

set -euo pipefail

echo "Fixing Docker iptables issues in WSL2..."
echo

# Update iptables to use legacy mode (works better in WSL2)
echo "Switching to iptables-legacy..."
sudo update-alternatives --set iptables /usr/sbin/iptables-legacy
sudo update-alternatives --set ip6tables /usr/sbin/ip6tables-legacy

# Load required kernel modules
echo "Loading required kernel modules..."
sudo modprobe bridge || true
sudo modprobe br_netfilter || true
sudo modprobe overlay || true

# Configure sysctl for Docker
echo "Configuring sysctl for Docker..."
sudo tee /etc/sysctl.d/99-docker.conf > /dev/null <<EOF
net.bridge.bridge-nf-call-iptables = 1
net.bridge.bridge-nf-call-ip6tables = 1
net.ipv4.ip_forward = 1
EOF

# Apply sysctl settings
sudo sysctl --system > /dev/null 2>&1

# Clean up any existing Docker iptables rules
echo "Cleaning up existing Docker iptables rules..."
sudo iptables -t nat -F DOCKER 2>/dev/null || true
sudo iptables -F DOCKER 2>/dev/null || true
sudo iptables -F DOCKER-ISOLATION-STAGE-1 2>/dev/null || true
sudo iptables -F DOCKER-ISOLATION-STAGE-2 2>/dev/null || true
sudo iptables -F DOCKER-USER 2>/dev/null || true

# Create Docker daemon configuration for WSL2
echo "Creating Docker daemon configuration for WSL2..."
sudo mkdir -p /etc/docker
sudo tee /etc/docker/daemon.json > /dev/null <<EOF
{
    "iptables": false,
    "bridge": "none",
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

# Stop any running Docker processes
echo "Stopping any existing Docker processes..."
sudo pkill dockerd 2>/dev/null || true
sudo pkill containerd 2>/dev/null || true
sleep 2

# Start Docker service
echo "Starting Docker service..."
sudo service docker start

# Wait for Docker to start
echo "Waiting for Docker daemon to start..."
count=0
while [ $count -lt 30 ]; do
    if sudo docker info > /dev/null 2>&1; then
        echo "✓ Docker daemon started successfully!"
        break
    fi
    sleep 1
    count=$((count + 1))
done

if [ $count -eq 30 ]; then
    echo "❌ Docker daemon failed to start"
    echo
    echo "Try running Docker manually with debugging:"
    echo "  sudo dockerd --debug"
    exit 1
fi

# Test Docker
echo
echo "Testing Docker..."
if sudo docker run --rm hello-world > /dev/null 2>&1; then
    echo "✓ Docker is working correctly!"
else
    echo "❌ Docker test failed"
    exit 1
fi

echo
echo "Docker is now running in WSL2!"
echo
echo "Remember to use 'sudo' with Docker commands, or run:"
echo "  newgrp docker"
echo
echo "To build the Windows executable:"
echo "  ./build-windows-docker.sh"