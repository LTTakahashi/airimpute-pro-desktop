#!/bin/bash
# Run AirImpute Pro with isolated library environment

set -e

echo "Starting AirImpute Pro with library isolation..."
echo "=============================================="

# Create isolated library directory
ISOLATED_DIR="$HOME/.airimpute-isolated"
mkdir -p "$ISOLATED_DIR/libs"

# Function to download and extract Ubuntu 22.04 packages
download_ubuntu_package() {
    local package=$1
    local filename=$2
    local url="http://archive.ubuntu.com/ubuntu/pool/main/${package:0:1}/${package}/${filename}"
    
    if [ ! -f "$ISOLATED_DIR/$filename" ]; then
        echo "Downloading $filename..."
        wget -q "$url" -O "$ISOLATED_DIR/$filename"
        dpkg-deb -x "$ISOLATED_DIR/$filename" "$ISOLATED_DIR/extracted/"
    fi
}

# Download required packages if not already present
if [ ! -f "$ISOLATED_DIR/.setup-complete" ]; then
    echo "Setting up isolated environment (one-time setup)..."
    mkdir -p "$ISOLATED_DIR/extracted"
    
    # Download Ubuntu 22.04 packages
    download_ubuntu_package "webkit2gtk" "libwebkit2gtk-4.0-37_2.44.0-0ubuntu0.22.04.1_amd64.deb"
    download_ubuntu_package "webkit2gtk" "libjavascriptcoregtk-4.0-18_2.44.0-0ubuntu0.22.04.1_amd64.deb"
    download_ubuntu_package "libsoup2.4" "libsoup2.4-1_2.74.2-3_amd64.deb"
    download_ubuntu_package "libsoup2.4" "libsoup-gnome2.4-1_2.74.2-3_amd64.deb"
    
    # Copy libraries
    cp -r "$ISOLATED_DIR/extracted/usr/lib/x86_64-linux-gnu/"* "$ISOLATED_DIR/libs/" 2>/dev/null || true
    
    touch "$ISOLATED_DIR/.setup-complete"
    echo "Setup complete!"
fi

# Create a wrapper that forces our libraries
cd "$(dirname "$0")"

# Build if needed
if [ ! -f "src-tauri/target/debug/airimpute-pro" ]; then
    echo "Binary not found, building..."
    npm run build
fi

# Set up isolated environment
export LD_LIBRARY_PATH="$ISOLATED_DIR/libs:$LD_LIBRARY_PATH"
export WEBKIT_DISABLE_COMPOSITING_MODE=1

# Use a more aggressive approach - create a namespace
if command -v unshare &> /dev/null; then
    echo "Using namespace isolation..."
    # Create new mount namespace and bind our libs
    exec unshare -r bash -c "
        export LD_LIBRARY_PATH='$ISOLATED_DIR/libs:$LD_LIBRARY_PATH'
        npm run tauri dev
    "
else
    # Fallback to regular execution
    echo "Running with LD_LIBRARY_PATH isolation..."
    npm run tauri dev
fi