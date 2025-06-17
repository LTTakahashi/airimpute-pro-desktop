#!/bin/bash
# Test script to verify webkit setup

echo "Testing webkit library setup..."

# Check for webkit 4.1 libraries
echo -e "\n1. Checking for webkit 4.1 libraries:"
find /usr/lib -name "libwebkit2gtk-4.1*" -type f 2>/dev/null | head -5
find /usr/lib -name "libjavascriptcoregtk-4.1*" -type f 2>/dev/null | head -5

# Check pkg-config
echo -e "\n2. Checking pkg-config for webkit:"
pkg-config --exists webkit2gtk-4.1 && echo "✓ webkit2gtk-4.1 found" || echo "✗ webkit2gtk-4.1 not found"
pkg-config --exists javascriptcoregtk-4.1 && echo "✓ javascriptcoregtk-4.1 found" || echo "✗ javascriptcoregtk-4.1 not found"

# Check Python
echo -e "\n3. Checking Python setup:"
python3 --version
echo "Python location: $(which python3)"
python3 -c "import sys; print(f'Python version: {sys.version}')"

# Check Rust/Cargo
echo -e "\n4. Checking Rust setup:"
cargo --version
rustc --version

# Test the tauri-dev.sh script environment setup
echo -e "\n5. Testing webkit workaround:"
WEBKIT_FIX_DIR="/tmp/webkit-test-$$"
mkdir -p "$WEBKIT_FIX_DIR"

# Find libraries
WEBKIT_41=$(find /usr/lib -name "libwebkit2gtk-4.1.so*" -type f | head -1)
JSCORE_41=$(find /usr/lib -name "libjavascriptcoregtk-4.1.so*" -type f | head -1)

if [ -n "$WEBKIT_41" ] && [ -n "$JSCORE_41" ]; then
    echo "✓ Found webkit 4.1 libraries:"
    echo "  - $WEBKIT_41"
    echo "  - $JSCORE_41"
    
    # Test symlink creation
    ln -sf "$WEBKIT_41" "$WEBKIT_FIX_DIR/libwebkit2gtk-4.0.so"
    ln -sf "$JSCORE_41" "$WEBKIT_FIX_DIR/libjavascriptcoregtk-4.0.so"
    
    if [ -L "$WEBKIT_FIX_DIR/libwebkit2gtk-4.0.so" ] && [ -L "$WEBKIT_FIX_DIR/libjavascriptcoregtk-4.0.so" ]; then
        echo "✓ Successfully created compatibility symlinks"
    else
        echo "✗ Failed to create symlinks"
    fi
else
    echo "✗ Could not find webkit 4.1 libraries"
fi

# Cleanup
rm -rf "$WEBKIT_FIX_DIR"

echo -e "\nWebkit setup test complete."