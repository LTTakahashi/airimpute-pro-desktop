#!/bin/bash

# Script to create symbolic links for webkit2gtk-4.0 pointing to 4.1
# This fixes the linking issue when only webkit2gtk-4.1 is available

echo "Creating symbolic links for webkit2gtk-4.0 -> webkit2gtk-4.1"

# Check if running with sufficient permissions
if [ "$EUID" -ne 0 ]; then 
    echo "This script needs to be run with sudo to create system library links"
    echo "Usage: sudo $0"
    exit 1
fi

# Create symbolic links for webkit2gtk
if [ -f "/usr/lib/x86_64-linux-gnu/libwebkit2gtk-4.1.so.0" ]; then
    ln -sf libwebkit2gtk-4.1.so.0 /usr/lib/x86_64-linux-gnu/libwebkit2gtk-4.0.so.37
    ln -sf libwebkit2gtk-4.0.so.37 /usr/lib/x86_64-linux-gnu/libwebkit2gtk-4.0.so
    echo "Created webkit2gtk-4.0 symlinks"
else
    echo "Error: libwebkit2gtk-4.1.so.0 not found"
    exit 1
fi

# Create symbolic links for javascriptcoregtk
if [ -f "/usr/lib/x86_64-linux-gnu/libjavascriptcoregtk-4.1.so.0" ]; then
    ln -sf libjavascriptcoregtk-4.1.so.0 /usr/lib/x86_64-linux-gnu/libjavascriptcoregtk-4.0.so.18
    ln -sf libjavascriptcoregtk-4.0.so.18 /usr/lib/x86_64-linux-gnu/libjavascriptcoregtk-4.0.so
    echo "Created javascriptcoregtk-4.0 symlinks"
else
    echo "Error: libjavascriptcoregtk-4.1.so.0 not found"
    exit 1
fi

echo "Symbolic links created successfully!"
echo ""
echo "You can now build your Tauri application."
echo ""
echo "To remove these symlinks later, run:"
echo "  sudo rm /usr/lib/x86_64-linux-gnu/libwebkit2gtk-4.0.so*"
echo "  sudo rm /usr/lib/x86_64-linux-gnu/libjavascriptcoregtk-4.0.so*"