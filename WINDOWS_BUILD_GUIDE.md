# Windows Cross-Compilation Guide for AirImpute Pro

## Overview

This guide describes how to build AirImpute Pro for Windows from a Linux environment using secure cross-compilation techniques.

## Build Methods

### Method 1: Docker Build (Recommended)

The Docker build provides a consistent, reproducible environment with all dependencies pre-installed.

```bash
# Run the Docker-based build
./build-windows-docker.sh
```

This will:
1. Build a secure Docker image with all dependencies
2. Run the cross-compilation process
3. Generate Windows installers in `dist/windows/`

### Method 2: Direct Build

If you have all dependencies installed locally:

```bash
# Run the build script directly
./build-windows-secure-v3.sh
```

Required dependencies:
- Rust with `x86_64-pc-windows-msvc` target
- Zig 0.11.0 (for cross-compilation)
- Node.js 18+ and npm
- NSIS (makensis command)
- osslsigncode (optional, for code signing)

## Security Features

The build process includes several security hardening measures:

### 1. Compiler Security Flags
- **ASLR** (Address Space Layout Randomization)
- **DEP** (Data Execution Prevention)
- **CFG** (Control Flow Guard)
- **High Entropy ASLR**
- **Intel CET** (Control-flow Enforcement Technology)
- **Stack Protection**

### 2. Dependency Verification
- SHA256 checksums for all downloaded dependencies
- Security audits that fail the build on vulnerabilities
- Pinned versions for reproducibility

### 3. Secure Configuration
- Restricted filesystem access scope
- Disabled dangerous permissions (shell.open, file deletion)
- Per-user installation (no admin rights required)
- WebView2 bootstrapper embedded

## Code Signing

To sign the Windows executable, provide the following environment variables:

```bash
export CODE_SIGNING_CERT_FILE="/path/to/certificate.pfx"
export CODE_SIGNING_KEY_FILE="/path/to/private-key.pem"
export CODE_SIGNING_KEY_PASSWORD="your-password"

./build-windows-docker.sh
```

## Build Output

The build generates:
- **NSIS Installer**: `AirImpute-Pro_1.0.0_x64-setup.exe`
- **Standalone Executable**: `airimpute-pro.exe`

Both files will be in:
- Docker build: `dist/windows/`
- Direct build: `build-windows/target/x86_64-pc-windows-msvc/release/bundle/nsis/`

## Troubleshooting

### Missing Dependencies

If the build fails due to missing dependencies, use the Docker build method or install:

```bash
# Ubuntu/Debian
sudo apt-get install -y nsis osslsigncode

# Install Zig
curl -L https://ziglang.org/download/0.11.0/zig-linux-x86_64-0.11.0.tar.xz | tar -xJ
export PATH="$PWD/zig-linux-x86_64-0.11.0:$PATH"

# Add Windows target to Rust
rustup target add x86_64-pc-windows-msvc
```

### WebView2 Runtime

The application requires Microsoft Edge WebView2 Runtime. The installer will:
1. Check if WebView2 is installed
2. Download and install it if needed (requires internet connection)
3. Proceed with the application installation

### Build Failures

1. **Linker errors**: Ensure Zig is properly installed and in PATH
2. **npm audit failures**: Update vulnerable dependencies or use `npm audit fix`
3. **Cargo audit failures**: Update Rust dependencies with security fixes

## CI/CD Integration

For GitHub Actions, use the provided workflow:

```yaml
- name: Build Windows Executable
  run: ./build-windows-secure-v3.sh
  env:
    TAURI_PRIVATE_KEY: ${{ secrets.TAURI_PRIVATE_KEY }}
    TAURI_KEY_PASSWORD: ${{ secrets.TAURI_KEY_PASSWORD }}
```

## Security Considerations

1. **Supply Chain**: All dependencies are verified with SHA256 checksums
2. **Build Environment**: Uses non-root user in Docker
3. **Runtime Security**: Executable includes all modern Windows security features
4. **Code Signing**: Strongly recommended for production releases

## Performance Notes

The release build is optimized with:
- Link-Time Optimization (LTO)
- Single codegen unit for maximum optimization
- Panic=abort for smaller binary size
- Strip debug symbols

## Testing the Build

After building, test on a Windows machine:

1. Copy the installer to Windows
2. Run the installer
3. Verify the application launches
4. Check Windows Defender doesn't flag it
5. Verify all features work correctly

## Support

For issues or questions:
- Check build logs in `build-windows/build-*.log`
- Review security scan results
- Ensure all dependencies are properly installed