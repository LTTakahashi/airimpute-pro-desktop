#!/bin/bash

echo "=== AirImpute Pro Desktop - Comprehensive Build Fix ==="
echo

# Detect OS
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    if grep -qi microsoft /proc/version; then
        OS="WSL2"
    else
        OS="Linux"
    fi
elif [[ "$OSTYPE" == "darwin"* ]]; then
    OS="macOS"
elif [[ "$OSTYPE" == "msys" || "$OSTYPE" == "cygwin" ]]; then
    OS="Windows"
else
    OS="Unknown"
fi

echo "Detected OS: $OS"
echo

cd src-tauri

# WSL2 specific fixes
if [ "$OS" == "WSL2" ]; then
    echo "=== WSL2 Build Options ==="
    echo "1. Quick build (disable HDF5) - Recommended for development"
    echo "2. Full build with dev-release profile (faster compilation)"
    echo "3. Full release build (slow, needs 12GB+ RAM)"
    echo
    read -p "Select option (1-3): " choice
    
    case $choice in
        1)
            echo "Building without HDF5 support..."
            cargo build --no-default-features --features "custom-protocol,python-support"
            ;;
        2)
            echo "Building with dev-release profile..."
            cargo build --profile dev-release
            ;;
        3)
            echo "WARNING: This requires 12GB+ RAM and may take 30+ minutes"
            echo "Make sure you've configured .wslconfig with enough memory!"
            read -p "Continue? (y/n): " confirm
            if [ "$confirm" == "y" ]; then
                cargo build --release
            fi
            ;;
        *)
            echo "Invalid option"
            exit 1
            ;;
    esac
fi

# Linux native
if [ "$OS" == "Linux" ]; then
    echo "Checking dependencies..."
    
    # Check for HDF5
    if ! pkg-config --exists hdf5; then
        echo "Missing HDF5. Install with: sudo apt install libhdf5-dev"
        echo "Or build without HDF5:"
        echo "  cargo build --no-default-features --features \"custom-protocol,python-support\""
        exit 1
    fi
    
    # Check for WebKit
    if pkg-config --exists webkit2gtk-4.1; then
        echo "Using webkit2gtk-4.1..."
        export PKG_CONFIG_PATH="/usr/lib/x86_64-linux-gnu/pkgconfig:$PKG_CONFIG_PATH"
    fi
    
    cargo build --release
fi

# macOS
if [ "$OS" == "macOS" ]; then
    echo "Building for macOS..."
    cargo build --release
fi

echo
echo "Build complete!"
echo "Executable location: target/release/airimpute-pro"