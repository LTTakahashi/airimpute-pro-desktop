#!/bin/bash
# Secure Windows build script for AirImpute Pro
# Implements Gemini's security recommendations

set -euo pipefail

echo "=============================================="
echo "AirImpute Pro - Secure Windows Build"
echo "=============================================="
echo ""

# Cleanup function to restore original files
cleanup() {
    echo ""
    echo "Restoring original configuration..."
    if [ -f "src-tauri/tauri.conf.json.bak" ]; then
        mv src-tauri/tauri.conf.json.bak src-tauri/tauri.conf.json
    fi
    if [ -f ".cargo/config.toml.bak" ]; then
        mv .cargo/config.toml.bak .cargo/config.toml
    fi
}
trap cleanup EXIT

# Step 1: Check prerequisites
echo "Step 1: Checking prerequisites..."
if ! command -v npm &> /dev/null; then
    echo "ERROR: npm is required but not installed."
    exit 1
fi

if ! command -v cargo &> /dev/null; then
    echo "ERROR: cargo is required but not installed."
    exit 1
fi

# Step 2: Prepare Windows-specific configuration
echo ""
echo "Step 2: Preparing Windows configuration..."

# Backup original files
cp src-tauri/tauri.conf.json src-tauri/tauri.conf.json.bak
if [ -f .cargo/config.toml ]; then
    cp .cargo/config.toml .cargo/config.toml.bak
fi

# Apply Windows-specific cargo config
cp .cargo/config-windows.toml .cargo/config.toml

# Step 3: Update tauri.conf.json for Windows security
echo ""
echo "Step 3: Securing Tauri configuration for Windows..."

# Use jq to modify tauri.conf.json if available, otherwise create a new one
if command -v jq &> /dev/null; then
    jq '.tauri.bundle.windows.nsis.installMode = "perUser" |
        .tauri.bundle.windows.timestampUrl = "http://timestamp.sectigo.com" |
        .tauri.allowlist.fs.scope = [
            "$APPDATA/AirImputePro/**",
            "$LOCALAPPDATA/AirImputePro/**", 
            "$DOCUMENT/AirImpute/**",
            "$TEMP/AirImputePro/**"
        ]' src-tauri/tauri.conf.json > src-tauri/tauri.conf.json.tmp
    mv src-tauri/tauri.conf.json.tmp src-tauri/tauri.conf.json
else
    echo "WARNING: jq not found. Using existing tauri.conf.json"
fi

# Step 4: Build frontend
echo ""
echo "Step 4: Building frontend..."
npm run build

# Step 5: Build Windows executable with Tauri
echo ""
echo "Step 5: Building Windows application..."

# Check if we should use cross-compilation
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    echo "Cross-compiling from Linux to Windows..."
    
    # Install Windows target if not already installed
    rustup target add x86_64-pc-windows-gnu || true
    
    # Build using Tauri with cross
    cd src-tauri
    cargo tauri build --target x86_64-pc-windows-gnu
    cd ..
else
    echo "Building natively on Windows..."
    npm run tauri build
fi

# Step 6: Display results
echo ""
echo "=============================================="
echo "Build complete!"
echo ""
echo "Windows installers location:"
echo "  MSI: src-tauri/target/release/bundle/msi/"
echo "  NSIS: src-tauri/target/release/bundle/nsis/"
echo ""
echo "IMPORTANT SECURITY NOTES:"
echo "1. The executable is unsigned - obtain a code signing certificate for production"
echo "2. Python embedding is not implemented - the app will fail without system Python"
echo "3. Consider using python-build-standalone for bundling Python"
echo ""
echo "To fix Python embedding:"
echo "1. Download python-build-standalone for Windows"
echo "2. Extract to src-tauri/python/"
echo "3. Update build.rs to set PYO3_PYTHON"
echo "4. Add python/ to tauri.conf.json resources"
echo "=============================================="