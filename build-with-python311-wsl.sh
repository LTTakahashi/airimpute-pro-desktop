#!/bin/bash

echo "=== Building AirImputePro on WSL with Python 3.11 ==="
echo

# Set working directory
cd "$(dirname "$0")"

# Detect if we're on WSL
if grep -qi microsoft /proc/version; then
    echo "WSL detected. Using system Python for build process."
    
    # Check for Python 3.11 on WSL
    if command -v python3.11 &> /dev/null; then
        export PYO3_PYTHON="python3.11"
        echo "Using system Python 3.11"
    elif command -v python3 &> /dev/null; then
        export PYO3_PYTHON="python3"
        PYTHON_VERSION=$(python3 --version 2>&1 | cut -d' ' -f2)
        echo "Using system Python $PYTHON_VERSION"
        echo "Warning: Python 3.11 is recommended. Install with:"
        echo "  sudo apt update && sudo apt install python3.11 python3.11-dev"
    else
        echo "Error: No Python found. Please install Python:"
        echo "  sudo apt update && sudo apt install python3.11 python3.11-dev"
        exit 1
    fi
    
    # For WSL builds, we don't use cross-compilation vars
    unset PYO3_CROSS_LIB_DIR
    unset PYO3_CROSS_PYTHON_VERSION
    unset PYO3_CROSS_PYTHON_IMPLEMENTATION
    unset PYO3_CROSS
    
else
    echo "Not on WSL. Using bundled Python for Windows build."
    
    # Original Windows build configuration
    export PYO3_PYTHON="$(pwd)/src-tauri/python/python.exe"
    export PYO3_CROSS_LIB_DIR="$(pwd)/src-tauri/python"
    export PYO3_CROSS_PYTHON_VERSION="3.11"
    export PYO3_CROSS_PYTHON_IMPLEMENTATION="CPython"
    export PYO3_CROSS="1"
fi

echo
echo "Environment variables:"
echo "  PYO3_PYTHON=$PYO3_PYTHON"
if [ -n "$PYO3_CROSS_LIB_DIR" ]; then
    echo "  PYO3_CROSS_LIB_DIR=$PYO3_CROSS_LIB_DIR"
    echo "  PYO3_CROSS_PYTHON_VERSION=$PYO3_CROSS_PYTHON_VERSION"
fi
echo

# Clean previous builds
echo "Cleaning previous builds..."
cd src-tauri
cargo clean
cd ..

# Build the application
echo "Building application..."
npm run tauri build

if [ $? -eq 0 ]; then
    echo
    echo "Build complete!"
    if grep -qi microsoft /proc/version; then
        echo "Note: You built on WSL. The output is a Linux executable."
        echo "To create a Windows executable, consider:"
        echo "  1. Building on native Windows"
        echo "  2. Using cross-compilation (see fix-wsl-python-build.sh)"
    else
        echo "Look for the output in src-tauri/target/release/bundle/"
    fi
else
    echo
    echo "Build failed. Common solutions:"
    echo "1. Install Python development files:"
    echo "   sudo apt install python3.11-dev"
    echo "2. Check error messages above"
    echo "3. Run ./fix-wsl-python-build.sh for more options"
fi