#!/bin/bash

# WSL-specific fix for libsoup conflict in Tauri applications
# This script handles the libsoup2/libsoup3 conflict when running Tauri on WSL

echo "Starting Tauri on WSL with libsoup conflict fix..."

# Method 1: Force WebKit to use a specific configuration
export WEBKIT_FORCE_SANDBOX=0
export WEBKIT_DISABLE_COMPOSITING_MODE=1

# Method 2: Use software rendering to avoid GPU-related issues in WSL
export LIBGL_ALWAYS_SOFTWARE=1

# Method 3: Force GTK to use X11 backend (more stable on WSL)
export GDK_BACKEND=x11

# Method 4: Disable hardware acceleration in Chromium/WebKit
export WEBKIT_DISABLE_DMABUF_RENDERER=1

# Method 5: Use a more compatible GTK theme
export GTK_THEME=Adwaita

# Method 6: Set display for X11 on WSL
export DISPLAY=:0

# Check if we're running on WSL
if grep -qi microsoft /proc/version; then
    echo "WSL detected. Applying WSL-specific fixes..."
    
    # Try to start X server if not running (for WSLg)
    if ! xset q &>/dev/null; then
        echo "X server not detected. Make sure WSLg is enabled or X server is running."
    fi
fi

echo "Environment variables set:"
echo "WEBKIT_FORCE_SANDBOX=$WEBKIT_FORCE_SANDBOX"
echo "WEBKIT_DISABLE_COMPOSITING_MODE=$WEBKIT_DISABLE_COMPOSITING_MODE"
echo "LIBGL_ALWAYS_SOFTWARE=$LIBGL_ALWAYS_SOFTWARE"
echo "GDK_BACKEND=$GDK_BACKEND"
echo "WEBKIT_DISABLE_DMABUF_RENDERER=$WEBKIT_DISABLE_DMABUF_RENDERER"
echo "GTK_THEME=$GTK_THEME"
echo "DISPLAY=$DISPLAY"

echo ""
echo "Launching Tauri development server..."
echo "If you still see libsoup errors, try running: sudo apt install libsoup2.4-dev"
echo ""

# Run Tauri with the fixed environment
npm run tauri dev