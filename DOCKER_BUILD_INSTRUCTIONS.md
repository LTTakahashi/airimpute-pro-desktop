# Windows Cross-Compilation Build Instructions

## Current Status
Docker is installed but the daemon needs to be started. The build environment is ready but requires Docker to be running.

## Step-by-Step Instructions

### 1. Start Docker Daemon
First, you need to start the Docker daemon:

```bash
sudo service docker start
```

If that doesn't work, try:
```bash
sudo dockerd
```

### 2. Verify Docker is Running
Check that Docker is working:

```bash
sudo docker info
```

### 3. Build the Docker Image
The Docker image needs to be built with the updated configuration:

```bash
sudo docker build \
    --file Dockerfile.windows-build \
    --tag airimpute-windows-builder:v1.0.0 \
    --build-arg ZIG_VERSION=0.11.0 \
    --build-arg ZIG_SHA256=2d00e789fec4f71790a6e7bf83ff91d564943c5ee843c5fd966efc474b423047 \
    .
```

### 4. Run the Windows Build
Once the Docker image is built, run the cross-compilation:

```bash
sudo docker run \
    --rm \
    --volume "$PWD:/build:rw" \
    --volume "$HOME/.cache/airimpute-build:/home/builder/.cache:rw" \
    --workdir /build \
    --user builder \
    airimpute-windows-builder:v1.0.0 \
    ./build-windows-secure-v3.sh
```

## Alternative: Run Everything with One Command
After starting Docker, you can use the provided script:

```bash
./build-windows-docker-sudo.sh
```

## Troubleshooting

### If Docker Won't Start
1. Check if it's a WSL issue:
   ```bash
   wsl --version
   ```

2. Try the iptables fix:
   ```bash
   sudo ./fix-docker-iptables.sh
   ```

### If Build Fails
1. Check the build log:
   ```bash
   cat build-windows/build-*.log
   ```

2. Test the Docker environment:
   ```bash
   ./test-docker-build.sh
   ```

### Permission Issues
If you get permission denied errors:

1. Add yourself to the docker group (already done)
2. Refresh your session:
   ```bash
   newgrp docker
   ```
   
   OR
   
3. Just use sudo for all Docker commands

## Expected Output
When successful, you'll find:
- `dist/windows/AirImpute-Pro_1.0.0_x64-setup.exe` - Windows installer
- `dist/windows/airimpute-pro.exe` - Standalone executable

## Current Issues Fixed
1. ✓ Python SHA256 updated to correct value
2. ✓ Rust toolchain configuration added to Dockerfile
3. ✓ Removed incompatible security audit tools
4. ✓ Docker networking configured for WSL2

## Next Steps
1. Start Docker daemon
2. Run the build script
3. The Windows executable will be created in the `dist/windows/` directory