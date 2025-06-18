# GitHub Actions Windows Build Guide

## Overview

This guide explains how to use GitHub Actions to build Windows executables for AirImpute Pro Desktop application. We provide two working methods: native Windows builds (recommended) and MinGW cross-compilation from Ubuntu.

## Prerequisites

1. GitHub repository with the workflow file
2. Optional: Tauri signing keys for auto-updater functionality
3. Optional: Windows code signing certificate

## Usage

### Manual Build Trigger

1. Go to your GitHub repository
2. Click on "Actions" tab
3. Select one of the workflows:
   - "Build Windows Executable (Native)" - Recommended
   - "Build Windows Executable (MinGW Cross-Compile)" - Alternative
4. Click "Run workflow"
5. Configure build parameters:
   - **Build type**: Choose between `release` (optimized) or `debug` (with debug symbols)
   - **Version**: Enter version number (e.g., 1.0.0)
6. Click "Run workflow" button

### Downloading Build Artifacts

After the workflow completes:

1. Go to the workflow run page
2. Scroll to "Artifacts" section
3. Download `windows-build-{type}-{version}` artifact
4. Extract the ZIP file to get:
   - `airimpute-pro.exe` - Main executable
   - `AirImputePro-Setup.exe` - NSIS installer
   - `checksums.txt` - SHA256 checksums for verification
   - `build-info.txt` - Detailed build information

## Build Process

The workflow performs these steps:

1. **Environment Setup**
   - Ubuntu 22.04 runner (stable environment)
   - Rust 1.82.0 with x86_64-pc-windows-msvc target
   - Node.js 20 for frontend
   - Python 3.10.11 for ML components
   - Zig 0.11.0 for cross-compilation
   - cargo-zigbuild 0.17.5 for efficient builds
   - NSIS for installer creation

2. **Dependency Installation**
   - System packages (build tools, SSL, NSIS)
   - npm for Node.js package management
   - Python packages from requirements.txt
   - Python Windows embeddable package download (for bundling)

3. **Cross-Compilation**
   - Frontend build with npm
   - Rust build using cargo-zigbuild
   - Targets x86_64-pc-windows-msvc (not GNU)
   - Uses Zig as the C/C++ compiler

4. **Installer Creation**
   - NSIS installer with:
     - Desktop shortcut creation
     - Start menu entries
     - Clean uninstaller
     - Program files installation

5. **Security & Verification**
   - SHA256 checksums generation
   - Optional code signing support
   - Build metadata tracking

## Security Configuration

### GitHub Secrets Setup

To enable auto-updater and code signing:

1. Go to Settings → Secrets and variables → Actions
2. Add the following secrets:
   - `TAURI_PRIVATE_KEY` - Tauri updater private key
   - `TAURI_KEY_PASSWORD` - Password for the private key
   - `WINDOWS_CERTIFICATE` - (Future) Code signing certificate

### Generating Tauri Keys

```bash
# Generate new key pair for Tauri updater
npm run tauri signer generate -- -w ~/.tauri/airimpute-pro.key

# The public key will be displayed - add it to tauri.conf.json
# The private key is in ~/.tauri/airimpute-pro.key
```

## Build Performance

- **First build**: ~20-25 minutes (downloading all dependencies)
- **Cached builds**: ~10-12 minutes
- **Cache validity**: Based on lock file changes
- **Artifact retention**: 30 days

### Cache Strategy

The workflow caches:
- Cargo registry and git dependencies
- Rust compilation artifacts
- npm cache
- pip packages

## Troubleshooting

### Common Issues

1. **Rust Compilation Errors**
   ```
   error: could not compile `package-name`
   ```
   - Ensure all dependencies support Windows MSVC target
   - Check for platform-specific code
   - Verify Cargo.toml has correct target configurations

2. **Zig Toolchain Issues**
   ```
   error: unable to find zig cc
   ```
   - The workflow installs Zig automatically
   - Verify the SHA256 checksum matches
   - Check Zig version compatibility

3. **NSIS Installer Failures**
   ```
   makensis: command not found
   ```
   - NSIS is installed via apt-get
   - Ensure the installer script syntax is correct

4. **Python Bundling Issues**
   - Python 3.10.11 embeddable AMD64 package is downloaded
   - SHA256 verification ensures correct version (608619f8619075629c9c69f361352a0da6ed7e62f83a0e19c63e0ea32eb7629d)
   - The embeddable package is extracted to python-embed-amd64 directory
   - Check requirements.txt compatibility

### Debugging Workflow Failures

1. Check the workflow logs in GitHub Actions
2. Look for the specific step that failed
3. Common failure points:
   - Dependency installation
   - Frontend build (npm/pnpm issues)
   - Rust compilation
   - NSIS installer creation

## Local Development

To replicate the CI environment locally:

```bash
# Use the Docker build environment
./build-windows-docker-with-retry.sh

# Or install dependencies manually
sudo apt-get update
sudo apt-get install -y build-essential curl wget pkg-config \
  libssl-dev nsis nsis-pluginapi lld llvm

# Install Zig
wget https://ziglang.org/download/0.11.0/zig-linux-x86_64-0.11.0.tar.xz
tar -xf zig-linux-x86_64-0.11.0.tar.xz
export PATH="$PWD/zig-linux-x86_64-0.11.0:$PATH"

# Install Rust with Windows target
rustup target add x86_64-pc-windows-msvc
cargo install cargo-zigbuild

# Build
cargo zigbuild --release --target x86_64-pc-windows-msvc
```

## Advanced Configuration

### Custom NSIS Scripts

Place custom NSIS scripts in `src-tauri/nsis/` directory:
- `installer.nsi` - Main installer script
- `uninstaller.nsi` - Custom uninstall logic

### Build Matrix

To build for multiple configurations, modify the workflow:

```yaml
strategy:
  matrix:
    build-type: [debug, release]
    version: ['1.0.0', '1.1.0']
```

### Release Automation

To automatically create releases on tags:

```yaml
on:
  push:
    tags:
      - 'v*'
```

## Maintenance

- **Toolchain Updates**: Update versions in workflow when new releases are available
- **Dependency Updates**: Keep lock files current
- **Security**: Regularly rotate signing keys
- **Monitoring**: Check GitHub Actions usage limits

## Differences from Previous Setup

This workflow uses:
- **x86_64-pc-windows-msvc** target instead of GNU
- **cargo-zigbuild** for more reliable cross-compilation
- **Ubuntu 22.04** for stability
- **Version parameter** for better release management
- **SHA256 checksums** for security verification

The MSVC target produces binaries that:
- Are compatible with more Windows systems
- Have better performance characteristics
- Work better with Windows development tools
- Support native Windows features