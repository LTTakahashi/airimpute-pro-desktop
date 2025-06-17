#!/bin/bash
# Run AirImpute Pro with forced libsoup2 loading
# This prevents libsoup3 from being loaded

set -e

echo "==================================================================="
echo "AirImpute Pro - Running with libsoup2 forced"
echo "==================================================================="
echo ""

# Find libsoup2
LIBSOUP2="/usr/lib/x86_64-linux-gnu/libsoup-2.4.so.1.11.2"
LIBSOUP_GNOME2="/usr/lib/x86_64-linux-gnu/libsoup-gnome-2.4.so.1.11.2"

if [ ! -f "$LIBSOUP2" ]; then
    echo "Error: libsoup2 not found at expected location"
    echo "Looking for alternatives..."
    LIBSOUP2=$(find /usr/lib* -name "libsoup-2.4.so*" -type f 2>/dev/null | head -1)
    if [ -z "$LIBSOUP2" ]; then
        echo "Error: Could not find libsoup2 on system"
        exit 1
    fi
fi

echo "Using libsoup2: $LIBSOUP2"

# Set environment to force libsoup2
export LD_PRELOAD="$LIBSOUP2:$LIBSOUP_GNOME2"

# Disable webkit compositing which can cause issues
export WEBKIT_DISABLE_COMPOSITING_MODE=1

# Force GI to use Soup 2.4
export GI_TYPELIB_PATH="/usr/lib/x86_64-linux-gnu/girepository-1.0"

# Set Python environment variables to prevent gi from loading soup3
export PYTHONDONTWRITEBYTECODE=1

echo ""
echo "Environment configured:"
echo "LD_PRELOAD=$LD_PRELOAD"
echo ""
echo "Starting application..."
echo ""

# Run with the existing tauri-dev.sh script that handles webkit compatibility
exec ./tauri-dev.sh "$@"