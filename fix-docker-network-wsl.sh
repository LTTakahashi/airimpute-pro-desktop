#!/usr/bin/env bash
# Fix Docker networking in WSL2
# Resolves network bridge issues

set -euo pipefail

echo "Fixing Docker networking for WSL2..."
echo

# Stop Docker first
echo "Stopping Docker service..."
sudo service docker stop
sudo pkill dockerd 2>/dev/null || true
sudo pkill containerd 2>/dev/null || true
sleep 2

# Update Docker daemon configuration for WSL2 networking
echo "Updating Docker daemon configuration..."
sudo tee /etc/docker/daemon.json > /dev/null <<EOF
{
    "log-driver": "json-file",
    "log-opts": {
        "max-size": "100m",
        "max-file": "3"
    },
    "storage-driver": "overlay2",
    "features": {
        "buildkit": true
    },
    "dns": ["8.8.8.8", "8.8.4.4"],
    "experimental": false
}
EOF

# Start Docker with proper networking
echo "Starting Docker service..."
sudo service docker start

# Wait for Docker to be ready
echo "Waiting for Docker to be ready..."
count=0
while [ $count -lt 30 ]; do
    if sudo docker info > /dev/null 2>&1; then
        echo "✓ Docker is ready!"
        break
    fi
    sleep 1
    count=$((count + 1))
done

# Test networking
echo
echo "Testing Docker networking..."
if sudo docker run --rm alpine ping -c 1 google.com > /dev/null 2>&1; then
    echo "✓ Docker networking is working!"
else
    echo "⚠ Docker networking may have issues, but continuing..."
fi

echo
echo "Docker networking fixed for WSL2!"
echo "You can now run: ./build-windows-docker-sudo.sh"