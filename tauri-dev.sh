#!/bin/bash
# Development script for AirImpute Pro with webkit 4.1 workaround

echo "Starting Tauri development server with webkit 4.1 support..."

# Create webkit redirect directory
WEBKIT_FIX_DIR="/tmp/webkit-fix-$$"
mkdir -p "$WEBKIT_FIX_DIR"

# Create pkg-config redirects
cat > "$WEBKIT_FIX_DIR/webkit2gtk-4.0.pc" << 'EOF'
Name: WebKit2GTK for GTK 3
Description: Web content engine for GTK applications
Version: 2.48.1

Requires: webkit2gtk-4.1
Libs: -lwebkit2gtk-4.1
Cflags:
EOF

cat > "$WEBKIT_FIX_DIR/javascriptcoregtk-4.0.pc" << 'EOF'
Name: JavaScriptCore for GTK
Description: JavaScript engine
Version: 2.48.1

Requires: javascriptcoregtk-4.1
Libs: -ljavascriptcoregtk-4.1
Cflags:
EOF

# Find webkit 4.1 libraries - use the .so symlink, not the versioned file
WEBKIT_41="/usr/lib/x86_64-linux-gnu/libwebkit2gtk-4.1.so"
JSCORE_41="/usr/lib/x86_64-linux-gnu/libjavascriptcoregtk-4.1.so"

if [ -f "$WEBKIT_41" ] && [ -f "$JSCORE_41" ]; then
    echo "Found webkit 4.1 libraries, creating compatibility symlinks..."
    # Create library symlinks with the exact names the linker expects
    ln -sf "$WEBKIT_41" "$WEBKIT_FIX_DIR/libwebkit2gtk-4.0.so"
    ln -sf "$JSCORE_41" "$WEBKIT_FIX_DIR/libjavascriptcoregtk-4.0.so"
    
    # Also create versioned symlinks that might be needed
    ln -sf "$WEBKIT_41.0" "$WEBKIT_FIX_DIR/libwebkit2gtk-4.0.so.37"
    ln -sf "$JSCORE_41.0" "$WEBKIT_FIX_DIR/libjavascriptcoregtk-4.0.so.18"
else
    echo "Warning: Could not find webkit 4.1 libraries!"
fi

# Set up environment
export PKG_CONFIG_PATH="$WEBKIT_FIX_DIR:$PKG_CONFIG_PATH"
export LD_LIBRARY_PATH="$WEBKIT_FIX_DIR:$LD_LIBRARY_PATH"
export LIBRARY_PATH="$WEBKIT_FIX_DIR:$LIBRARY_PATH"
export PYO3_PYTHON=/usr/bin/python3

# Add the webkit fix directory to the linker search path
export RUSTFLAGS="-C link-arg=-L$WEBKIT_FIX_DIR"

# Also tell the linker to look in the standard lib directory
export RUSTFLAGS="$RUSTFLAGS -C link-arg=-L/usr/lib/x86_64-linux-gnu"

# Debug environment
echo "--- DEBUGGING WEBKIT SETUP ---"
echo "WEBKIT_FIX_DIR: $WEBKIT_FIX_DIR"
echo "PKG_CONFIG_PATH: $PKG_CONFIG_PATH"
echo "LD_LIBRARY_PATH: $LD_LIBRARY_PATH"
echo "LIBRARY_PATH: $LIBRARY_PATH"
echo "Testing pkg-config for webkit2gtk-4.0:"
pkg-config --cflags --libs webkit2gtk-4.0 2>&1 || echo "Failed"
echo "Testing pkg-config for webkit2gtk-4.1:"
pkg-config --cflags --libs webkit2gtk-4.1 2>&1 || echo "Failed"
echo "--- END DEBUG ---"

# Clean up function
cleanup() {
    echo "Cleaning up webkit fix directory..."
    rm -rf "$WEBKIT_FIX_DIR"
}
trap cleanup EXIT

# Run tauri dev
echo "Starting Tauri development server..."
# Ensure the environment is passed to npm/cargo
exec npm run tauri dev "$@"