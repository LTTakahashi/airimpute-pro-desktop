# LibSoup Conflict Resolution

## Problem

The application encounters a runtime conflict between libsoup2 and libsoup3:
```
libsoup3 symbols detected. Using libsoup2 and libsoup3 in the same process is not supported.
```

This occurs because:
- Tauri v1 depends on webkit2gtk-rs v0.18, which uses libsoup2
- Modern Linux systems (Ubuntu 22.04+, Fedora 36+) use webkit2gtk-4.1 with libsoup3
- Both libraries cannot coexist in the same process

## Solutions

### 1. Immediate Solution: Nix Shell (Recommended for Development)

We've created a Nix shell environment that provides the correct webkit version:

```bash
# Install Nix if not already installed
sh <(curl -L https://nixos.org/nix/install) --daemon

# Run the app with Nix
./run-with-nix.sh
```

This creates an isolated environment with webkit2gtk-4.0 and libsoup2.

### 2. Docker Solution (Alternative)

Use the provided Docker setup:

```bash
# With Docker installed
./docker-dev.sh
```

### 3. Long-term Solution: Upgrade to Tauri v2

The permanent fix is to migrate to Tauri v2, which supports modern webkit:

```bash
cargo tauri migrate
# Follow the migration guide
```

## Current Status

- ✅ Build issues resolved (PyO3 configuration fixed)
- ✅ Python integration configured correctly
- ✅ Compilation successful
- ⚠️ Runtime requires isolated environment (Nix or Docker)
- 📋 255 warnings (non-critical, mostly unused code)

## Quick Start

For immediate development:
```bash
./run-with-nix.sh
```

This will automatically:
1. Check for Nix installation
2. Set up the correct environment
3. Run the application with compatible libraries