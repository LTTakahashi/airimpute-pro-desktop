#!/bin/bash
# Build AppImage with webkit 4.0 compatibility

set -e

echo "Building AirImpute Pro AppImage..."
echo "=================================="

# Check if we're on a system with the right libraries
if [ ! -f "/usr/lib/x86_64-linux-gnu/libwebkit2gtk-4.0.so" ]; then
    echo "Warning: libwebkit2gtk-4.0 not found on this system"
    echo "The AppImage might not bundle correctly"
fi

# Install AppImage tools if not present
if ! command -v appimagetool &> /dev/null; then
    echo "Installing appimagetool..."
    wget -q https://github.com/AppImage/AppImageKit/releases/download/continuous/appimagetool-x86_64.AppImage -O /tmp/appimagetool
    chmod +x /tmp/appimagetool
    APPIMAGETOOL="/tmp/appimagetool"
else
    APPIMAGETOOL="appimagetool"
fi

# Build the application
echo "Building release version..."
npm run build

# Build Tauri with AppImage target
echo "Building AppImage..."
npm run tauri build -- --target appimage

echo ""
echo "Build complete! The AppImage should be in:"
echo "  src-tauri/target/release/bundle/appimage/"
echo ""
echo "This AppImage includes bundled libsoup2 libraries and should work on systems with libsoup3."