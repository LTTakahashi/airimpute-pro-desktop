#!/bin/bash
# Run AirImpute Pro with bundled libraries workaround

set -e

echo "Starting AirImpute Pro with bundled library workaround..."
echo "========================================================"

# Create a local lib directory
BUNDLE_DIR="$HOME/.airimpute-bundled-libs"
mkdir -p "$BUNDLE_DIR"

# Check if we need to download libraries
if [ ! -f "$BUNDLE_DIR/.libs-installed" ]; then
    echo "Setting up bundled libraries..."
    echo "This is a one-time setup process."
    echo ""
    
    # Create a temporary directory for downloads
    TEMP_DIR=$(mktemp -d)
    cd "$TEMP_DIR"
    
    echo "Downloading compatible webkit libraries from Ubuntu 20.04..."
    
    # Download packages (these URLs are for Ubuntu 20.04 focal)
    wget -q http://archive.ubuntu.com/ubuntu/pool/main/w/webkit2gtk/libwebkit2gtk-4.0-37_2.38.6-0ubuntu0.20.04.1_amd64.deb
    wget -q http://archive.ubuntu.com/ubuntu/pool/main/w/webkit2gtk/libjavascriptcoregtk-4.0-18_2.38.6-0ubuntu0.20.04.1_amd64.deb
    wget -q http://archive.ubuntu.com/ubuntu/pool/main/libs/libsoup2.4/libsoup2.4-1_2.70.0-1_amd64.deb
    wget -q http://archive.ubuntu.com/ubuntu/pool/main/libs/libsoup2.4/libsoup-gnome2.4-1_2.70.0-1_amd64.deb
    
    echo "Extracting libraries..."
    for deb in *.deb; do
        dpkg-deb -x "$deb" extracted/
    done
    
    # Copy libraries to bundle directory
    cp -P extracted/usr/lib/x86_64-linux-gnu/libwebkit2gtk-4.0.so* "$BUNDLE_DIR/"
    cp -P extracted/usr/lib/x86_64-linux-gnu/libjavascriptcoregtk-4.0.so* "$BUNDLE_DIR/"
    cp -P extracted/usr/lib/x86_64-linux-gnu/libsoup-2.4.so* "$BUNDLE_DIR/"
    cp -P extracted/usr/lib/x86_64-linux-gnu/libsoup-gnome-2.4.so* "$BUNDLE_DIR/"
    
    # Mark as installed
    touch "$BUNDLE_DIR/.libs-installed"
    
    # Cleanup
    cd -
    rm -rf "$TEMP_DIR"
    
    echo "Library setup complete!"
fi

# Return to app directory
cd "$(dirname "$0")"

# Set up environment
export LD_LIBRARY_PATH="$BUNDLE_DIR:$LD_LIBRARY_PATH"
export WEBKIT_DISABLE_COMPOSITING_MODE=1

echo ""
echo "Starting application with bundled libraries..."
echo "Note: This is a temporary workaround. For production use,"
echo "please consider upgrading to Tauri v2."
echo ""

# Run the app
npm run tauri dev