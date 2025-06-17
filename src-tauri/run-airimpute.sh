#!/bin/bash
# Wrapper script to run AirImpute Pro with proper environment

# Set webkit environment to avoid libsoup conflicts
export WEBKIT_DISABLE_COMPOSITING_MODE=1

# Run with GDK backend set to X11 to avoid Wayland issues in WSL
export GDK_BACKEND=x11

# Set display for WSL
export DISPLAY=${DISPLAY:-:0}

# Navigate to the correct directory
cd "$(dirname "$0")"

# Run the application
exec ./target/debug/airimpute-pro "$@"