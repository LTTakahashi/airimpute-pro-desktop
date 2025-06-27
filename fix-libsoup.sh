#!/bin/bash

# Fix libsoup2/libsoup3 conflict for Tauri applications

echo "Fixing libsoup conflict for Tauri application..."

# Method 1: Use environment variables to force webkit to use specific version
export WEBKIT_FORCE_SANDBOX=0
export WEBKIT_DISABLE_COMPOSITING_MODE=1
export GDK_BACKEND=x11

# Method 2: Preload libsoup2 to prevent libsoup3 from loading
# Find libsoup2 library path
LIBSOUP2_PATH=$(find /usr/lib -name "libsoup-2.4.so.1" 2>/dev/null | head -1)

if [ -n "$LIBSOUP2_PATH" ]; then
    echo "Found libsoup2 at: $LIBSOUP2_PATH"
    export LD_PRELOAD=$LIBSOUP2_PATH
else
    echo "Warning: Could not find libsoup2 library"
fi

# Method 3: Alternative - use GTK_USE_PORTAL to avoid the conflict
export GTK_USE_PORTAL=1

echo "Environment variables set:"
echo "WEBKIT_FORCE_SANDBOX=$WEBKIT_FORCE_SANDBOX"
echo "WEBKIT_DISABLE_COMPOSITING_MODE=$WEBKIT_DISABLE_COMPOSITING_MODE"
echo "GDK_BACKEND=$GDK_BACKEND"
echo "GTK_USE_PORTAL=$GTK_USE_PORTAL"
[ -n "$LD_PRELOAD" ] && echo "LD_PRELOAD=$LD_PRELOAD"

echo ""
echo "Starting Tauri dev server..."
npm run tauri dev