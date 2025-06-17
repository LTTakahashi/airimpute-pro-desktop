#!/bin/bash
# Debug script to investigate the libsoup conflict

echo "Debugging libsoup conflict for AirImpute Pro..."
echo "=============================================="

# First, let's see what libraries the built binary links to
BINARY_PATH="./src-tauri/target/debug/airimpute-pro"

if [ -f "$BINARY_PATH" ]; then
    echo "Checking binary dependencies:"
    ldd "$BINARY_PATH" | grep -E "soup|webkit" || echo "No soup/webkit dependencies found"
    echo ""
fi

# Check environment
echo "Current environment:"
echo "LD_LIBRARY_PATH: $LD_LIBRARY_PATH"
echo "PKG_CONFIG_PATH: $PKG_CONFIG_PATH"
echo ""

# Try to run with debug output
echo "Attempting to run with detailed debug output..."
export RUST_BACKTRACE=full
export RUST_LOG=debug
export G_MESSAGES_DEBUG=all
export WEBKIT_DEBUG=all

# Try to preload libsoup2 explicitly
echo "Trying to run with explicit libsoup2 preload..."
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libsoup-2.4.so.1

# Run the app
npm run tauri dev 2>&1 | tee debug-output.log