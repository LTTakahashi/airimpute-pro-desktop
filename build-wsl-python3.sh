#!/bin/bash

# Build script for WSL using system Python 3

echo "Building Tauri app on WSL with system Python..."

# Check if we're on WSL
if [[ ! $(uname -r) =~ Microsoft|WSL ]]; then
    echo "Warning: This doesn't appear to be WSL."
fi

# Get Python info
PYTHON_CMD="python3"
PYTHON_VERSION=$($PYTHON_CMD --version 2>&1)
echo "Using system Python: $PYTHON_VERSION"

# Set environment variables for pyo3
export PYO3_PYTHON=$PYTHON_CMD

# Change to the src-tauri directory
cd src-tauri || exit 1

# Clean previous builds
echo "Cleaning previous build artifacts..."
cargo clean

# Build the Tauri app
echo "Running cargo build..."
cargo build --release

# Check if the build was successful
if [ $? -eq 0 ]; then
    echo ""
    echo "Build successful!"
    echo "The Linux executable is at: target/release/airimpute-pro-desktop"
    echo ""
    echo "Note: This is a Linux build for WSL. To create a Windows build:"
    echo "1. Build on native Windows (not WSL)"
    echo "2. Or use the Windows build scripts (.bat files)"
else
    echo ""
    echo "Build failed. The error above indicates the issue."
fi