# Docker Network Bridge Error - Troubleshooting Guide

## The Error
```
ERROR: failed to solve: process "/bin/sh -c apt-get update..." did not complete successfully: network bridge not found
```

This is a common issue in WSL2 when Docker's network bridge is missing or corrupted.

## Quick Fix Attempts

### 1. Use the WSL2 Docker Start Script
```bash
sudo ./start-docker-wsl2.sh
```
This script:
- Stops any existing Docker daemon
- Configures iptables for WSL2
- Starts Docker with WSL2-specific options
- Creates the bridge network if missing

### 2. Fix Docker Network
```bash
sudo ./fix-docker-network.sh
```
This script:
- Cleans up corrupted network files
- Restarts Docker with debug logging
- Recreates the default bridge

### 3. Build with Retry and Host Network
```bash
./build-windows-docker-with-retry.sh
```
This script:
- Uses `--network=host` to bypass bridge issues
- Retries the build up to 3 times
- Attempts to fix networking between retries

## Manual Troubleshooting Steps

### Step 1: Check Docker Status
```bash
sudo docker version
sudo docker info
```

### Step 2: Check Networks
```bash
sudo docker network ls
```

If "bridge" network is missing:
```bash
sudo docker network create bridge
```

### Step 3: Clean Docker Networks
```bash
# Stop Docker
sudo service docker stop

# Remove network files
sudo rm -rf /var/lib/docker/network/files/*

# Start Docker
sudo service docker start
```

### Step 4: Use Alternative iptables
```bash
# Switch to iptables-legacy (more compatible with WSL2)
sudo update-alternatives --set iptables /usr/sbin/iptables-legacy
sudo update-alternatives --set ip6tables /usr/sbin/ip6tables-legacy

# Restart Docker
sudo service docker restart
```

### Step 5: Start Docker with Minimal Options
```bash
# Stop existing daemon
sudo service docker stop
sudo killall dockerd 2>/dev/null || true

# Start with minimal networking
sudo dockerd --iptables=false --bridge=none &

# Wait a moment
sleep 5

# Test
sudo docker run --rm hello-world
```

## Alternative Solutions

### Option 1: Use Docker Desktop for Windows
If WSL2 Docker continues to have issues, consider installing Docker Desktop for Windows, which handles WSL2 integration automatically.

### Option 2: Build Without Network During Image Creation
Modify the Dockerfile to download dependencies in a separate step:

```dockerfile
# Download stage
FROM alpine as downloader
RUN wget https://... -O /downloads/file.tar.gz

# Build stage
FROM debian:bookworm-slim
COPY --from=downloader /downloads /downloads
# Install from local files...
```

### Option 3: Use Host Network for Build
The `build-windows-docker-with-retry.sh` script already does this:
```bash
docker build --network=host ...
```

## WSL2-Specific Issues

### DNS Resolution
If DNS isn't working in containers:
```bash
# Check WSL2 DNS
cat /etc/resolv.conf

# Add Google DNS to Docker
sudo mkdir -p /etc/docker
echo '{"dns": ["8.8.8.8", "8.8.4.4"]}' | sudo tee /etc/docker/daemon.json
sudo service docker restart
```

### Firewall/VPN Interference
Some VPNs or corporate firewalls can interfere with Docker in WSL2:
- Temporarily disconnect from VPN
- Check Windows Firewall settings
- Try with Windows Defender disabled temporarily

## Verification

After fixing, test with:
```bash
# Test basic connectivity
sudo docker run --rm alpine ping -c 1 google.com

# Test apt-get
sudo docker run --rm debian:bookworm-slim apt-get update

# Run network test
./test-docker-network.sh
```

## If All Else Fails

1. **Restart WSL2**:
   ```powershell
   # In PowerShell as Administrator
   wsl --shutdown
   # Then restart your WSL2 terminal
   ```

2. **Reset Docker**:
   ```bash
   sudo apt-get purge docker-ce docker-ce-cli containerd.io
   sudo rm -rf /var/lib/docker
   sudo apt-get install docker-ce docker-ce-cli containerd.io
   ```

3. **Use the Retry Script**:
   The `build-windows-docker-with-retry.sh` script will attempt multiple times and try to fix issues between attempts.

## Current Status
- Dockerfile has been updated to remove version constraints
- Build scripts use `--network=host` to bypass bridge issues
- Multiple recovery scripts are available
- Ready to build once Docker networking is fixed