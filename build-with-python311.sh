#!/bin/bash

echo "=== Building AirImputePro with Python 3.11 ==="
echo

# Set working directory
cd "$(dirname "$0")"

# First, ensure we have the python310.dll compatibility link
if [ -f "src-tauri/python/python311.dll" ] && [ ! -f "src-tauri/python/python310.dll" ]; then
    echo "Creating python310.dll symlink for compatibility..."
    cp src-tauri/python/python311.dll src-tauri/python/python310.dll
fi

# Set environment variables for PyO3
export PYO3_PYTHON="$(pwd)/src-tauri/python/python.exe"
export PYO3_CROSS_LIB_DIR="$(pwd)/src-tauri/python"
export PYO3_CROSS_PYTHON_VERSION="3.11"

# For Windows builds, also set these
export PYO3_CROSS_PYTHON_IMPLEMENTATION="CPython"
export PYO3_CROSS="1"

echo "Environment variables set:"
echo "  PYO3_PYTHON=$PYO3_PYTHON"
echo "  PYO3_CROSS_LIB_DIR=$PYO3_CROSS_LIB_DIR"
echo "  PYO3_CROSS_PYTHON_VERSION=$PYO3_CROSS_PYTHON_VERSION"
echo

# Clean previous builds
echo "Cleaning previous builds..."
cd src-tauri
cargo clean
cd ..

# Build the application
echo "Building application..."
npm run tauri build

echo
echo "Build complete! The executable should now work with the bundled Python."
echo "Look for the output in src-tauri/target/release/bundle/"