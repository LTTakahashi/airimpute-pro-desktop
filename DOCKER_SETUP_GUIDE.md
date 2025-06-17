# Docker Setup Guide for Windows Cross-Compilation

## Quick Start

Run the automated setup script:

```bash
./setup-docker-build-env.sh
```

This script will:
1. Check if Docker is installed
2. Install Docker if needed (requires sudo password)
3. Configure Docker for your user
4. Build the Windows cross-compilation Docker image
5. Test the environment

## Manual Installation

If you prefer to install Docker manually:

### 1. Install Docker

```bash
# Run the Docker installation script
./install-docker.sh
```

Or follow the [official Docker installation guide for Ubuntu](https://docs.docker.com/engine/install/ubuntu/).

### 2. Configure User Permissions

After installation, add yourself to the docker group:

```bash
sudo usermod -aG docker $USER
```

Then refresh your session:

```bash
# Option 1: Quick refresh for current terminal
newgrp docker

# Option 2: Log out and log back in (for all sessions)
```

### 3. Verify Installation

Test that Docker works without sudo:

```bash
docker run hello-world
```

### 4. Build the Cross-Compilation Image

```bash
# Build the Docker image
docker build \
    --file Dockerfile.windows-build \
    --tag airimpute-windows-builder:v1.0.0 \
    .
```

## Building Windows Executable

Once Docker is set up, build the Windows executable:

```bash
./build-windows-docker.sh
```

The build output will be in `dist/windows/`:
- `AirImpute-Pro_1.0.0_x64-setup.exe` - NSIS installer
- `airimpute-pro.exe` - Standalone executable

## Troubleshooting

### Permission Denied

If you get "permission denied" when running Docker:

```bash
# Check if you're in the docker group
groups

# If docker is not listed, add yourself
sudo usermod -aG docker $USER

# Refresh your session
newgrp docker
```

### Docker Daemon Not Running

If Docker daemon is not running:

```bash
# Start Docker
sudo systemctl start docker

# Enable Docker to start on boot
sudo systemctl enable docker

# Check status
sudo systemctl status docker
```

### Build Failures

If the build fails:

1. Check Docker logs:
   ```bash
   docker logs <container-id>
   ```

2. Run build with more verbosity:
   ```bash
   docker run --rm -it \
       --volume "$PWD:/build:rw" \
       --workdir /build \
       --user builder \
       airimpute-windows-builder:v1.0.0 \
       bash
   # Then run build commands manually
   ```

3. Check available disk space:
   ```bash
   df -h
   docker system df
   ```

4. Clean up Docker resources:
   ```bash
   docker system prune -a
   ```

## Security Notes

The Docker setup includes:
- Non-root user (`builder`) in container
- Minimal base image (debian slim)
- Pinned package versions
- SHA256 verification for downloads
- Read-only container filesystem (where possible)

## Advanced Configuration

### Using Different Cache Directory

```bash
# Set custom cache directory
export AIRIMPUTE_CACHE_DIR=/path/to/cache
./build-windows-docker.sh
```

### Building with Code Signing

```bash
# Set signing credentials
export CODE_SIGNING_CERT_FILE=/path/to/cert.pfx
export CODE_SIGNING_KEY_PASSWORD="your-password"
./build-windows-docker.sh
```

### Using Proxy

If behind a corporate proxy:

```bash
# Configure Docker to use proxy
sudo mkdir -p /etc/systemd/system/docker.service.d
sudo tee /etc/systemd/system/docker.service.d/http-proxy.conf <<EOF
[Service]
Environment="HTTP_PROXY=http://proxy.example.com:8080"
Environment="HTTPS_PROXY=http://proxy.example.com:8080"
Environment="NO_PROXY=localhost,127.0.0.1"
EOF

sudo systemctl daemon-reload
sudo systemctl restart docker
```

## Maintenance

### Update Docker

```bash
sudo apt-get update
sudo apt-get upgrade docker-ce docker-ce-cli containerd.io
```

### Rebuild Docker Image

When dependencies change:

```bash
# Force rebuild without cache
docker build --no-cache \
    --file Dockerfile.windows-build \
    --tag airimpute-windows-builder:v1.0.0 \
    .
```

### Clean Up Old Images

```bash
# Remove unused images
docker image prune -a

# Remove all stopped containers
docker container prune

# Full cleanup (careful!)
docker system prune -a --volumes
```