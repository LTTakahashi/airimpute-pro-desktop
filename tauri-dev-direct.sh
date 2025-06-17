#!/bin/bash
# Direct Tauri development script that bypasses npm

echo "Starting Tauri development server with direct cargo invocation..."

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

# Find webkit 4.1 libraries
WEBKIT_41=$(find /usr/lib -name "libwebkit2gtk-4.1.so*" -type f | head -1)
JSCORE_41=$(find /usr/lib -name "libjavascriptcoregtk-4.1.so*" -type f | head -1)

if [ -n "$WEBKIT_41" ] && [ -n "$JSCORE_41" ]; then
    # Create library symlinks
    ln -sf "$WEBKIT_41" "$WEBKIT_FIX_DIR/libwebkit2gtk-4.0.so"
    ln -sf "$JSCORE_41" "$WEBKIT_FIX_DIR/libjavascriptcoregtk-4.0.so"
fi

# Set up environment
export PKG_CONFIG_PATH="$WEBKIT_FIX_DIR:$PKG_CONFIG_PATH"
export LD_LIBRARY_PATH="$WEBKIT_FIX_DIR:$LD_LIBRARY_PATH"
export LIBRARY_PATH="$WEBKIT_FIX_DIR:$LIBRARY_PATH"
export PYO3_PYTHON=/usr/bin/python3
export RUSTFLAGS="-C link-arg=-lpython3.12"

# Debug environment
echo "--- DEBUGGING WEBKIT SETUP ---"
echo "WEBKIT_FIX_DIR: $WEBKIT_FIX_DIR"
echo "PKG_CONFIG_PATH: $PKG_CONFIG_PATH"
echo "Testing pkg-config:"
pkg-config --cflags --libs webkit2gtk-4.0 2>&1 | head -1
echo "--- END DEBUG ---"

# Clean up function
cleanup() {
    echo "Cleaning up..."
    # Kill the frontend server if running
    if [ ! -z "$VITE_PID" ]; then
        kill $VITE_PID 2>/dev/null || true
    fi
    rm -rf "$WEBKIT_FIX_DIR"
}
trap cleanup EXIT INT TERM

# Start frontend server in background
echo "Starting frontend server..."
cd .. && npm run dev &
VITE_PID=$!
cd src-tauri

# Give frontend time to start
sleep 2

# Run cargo tauri dev directly
echo "Starting Rust backend..."
cargo tauri dev "$@"