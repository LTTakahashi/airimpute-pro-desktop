#!/usr/bin/env bash
# Test Docker networking

echo "=== Docker Network Test ==="

# Check Docker daemon
if ! sudo docker version &>/dev/null; then
    echo "❌ Docker daemon not running"
    echo "Try: sudo ./fix-docker-network.sh"
    exit 1
fi

echo "✓ Docker daemon is running"

# List networks
echo
echo "Docker networks:"
sudo docker network ls

# Test network connectivity
echo
echo "Testing network connectivity..."
if sudo docker run --rm alpine ping -c 1 google.com &>/dev/null; then
    echo "✓ Docker can access internet"
else
    echo "❌ Docker cannot access internet"
fi

# Test DNS
echo
echo "Testing DNS resolution..."
if sudo docker run --rm alpine nslookup google.com &>/dev/null; then
    echo "✓ DNS resolution works"
else
    echo "❌ DNS resolution failed"
fi

# Try a simple apt-get update
echo
echo "Testing apt-get in container..."
if sudo docker run --rm debian:bookworm-slim apt-get update &>/dev/null; then
    echo "✓ apt-get update works"
else
    echo "❌ apt-get update failed"
    echo
    echo "This might be due to:"
    echo "1. Docker network bridge issues"
    echo "2. WSL2 DNS configuration"
    echo "3. Firewall/proxy settings"
fi

echo
echo "=== Test Complete ==="