#!/usr/bin/env bash
# Simple Docker starter for WSL
# Run this to start Docker daemon

set -euo pipefail

echo "Starting Docker daemon..."
echo "You'll need to enter your sudo password."
echo

# Check if Docker daemon is already running
if pgrep -x dockerd > /dev/null; then
    echo "✓ Docker daemon is already running!"
    docker version
    exit 0
fi

# Start Docker daemon
echo "Starting Docker service..."
sudo service docker start

# Wait a moment for it to fully start
sleep 2

# Check if it started successfully
if sudo docker info > /dev/null 2>&1; then
    echo "✓ Docker daemon started successfully!"
    echo
    
    # Add user to docker group if needed
    if ! groups | grep -q docker; then
        echo "Adding $USER to docker group..."
        sudo usermod -aG docker "$USER"
        echo
        echo "IMPORTANT: Run 'newgrp docker' to use Docker without sudo"
        echo "Or log out and back in."
    else
        echo "You're already in the docker group."
        echo "If Docker still requires sudo, run: newgrp docker"
    fi
    
    echo
    echo "Docker is ready! You can now run:"
    echo "  ./build-windows-docker.sh"
else
    echo "❌ Failed to start Docker daemon"
    echo
    echo "Try running manually:"
    echo "  sudo dockerd"
    exit 1
fi