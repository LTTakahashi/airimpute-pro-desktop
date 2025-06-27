#!/bin/bash

# Build script for WSL environment
# This script uses the system Python instead of the bundled Windows Python

echo "Building Tauri app on WSL..."

# Check if we're on WSL
if [[ ! $(uname -r) =~ Microsoft|WSL ]]; then
    echo "Warning: This doesn't appear to be WSL. Use build-with-python311.sh instead."
    exit 1
fi

# Check for Python 3.11
if ! command -v python3.11 &> /dev/null; then
    echo "Error: Python 3.11 is not installed."
    echo "Please install it with: sudo apt update && sudo apt install python3.11 python3.11-dev"
    exit 1
fi

# Get Python info
PYTHON_VERSION=$(python3.11 --version | cut -d' ' -f2)
echo "Using system Python: $PYTHON_VERSION"

# Set environment variables to use system Python for pyo3
export PYO3_PYTHON=python3.11

# Change to the src-tauri directory
cd src-tauri || exit 1

# Build the Tauri app
echo "Running cargo build..."
cargo build --release

# Check if the build was successful
if [ $? -eq 0 ]; then
    echo ""
    echo "Build successful!"
    echo "Note: This is a Linux build. To create a Windows build, you need to:"
    echo "1. Build on native Windows (not WSL), or"
    echo "2. Use cross-compilation tools"
else
    echo ""
    echo "Build failed. Common solutions:"
    echo "1. Make sure Python 3.11 dev packages are installed:"
    echo "   sudo apt install python3.11-dev"
    echo "2. Try clearing the cargo cache:"
    echo "   cargo clean"
    echo "3. Check the error messages above for specific issues"
fi