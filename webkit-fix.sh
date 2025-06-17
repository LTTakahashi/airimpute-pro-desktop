#!/bin/bash
# WebKit compatibility fix for development
# This script creates a local environment that prioritizes libsoup2

set -e

echo "Setting up webkit compatibility environment..."

# Create a local lib directory
WEBKIT_FIX_DIR="$HOME/.airimpute-webkit-fix"
mkdir -p "$WEBKIT_FIX_DIR"

# Check if we have libsoup2 installed
if [ ! -f "/usr/lib/x86_64-linux-gnu/libsoup-2.4.so.1" ]; then
    echo "Error: libsoup2 not found on this system"
    echo "Please install it with: sudo apt install libsoup2.4-1"
    exit 1
fi

# Create symlinks to force libsoup2 usage
ln -sf /usr/lib/x86_64-linux-gnu/libsoup-2.4.so.1 "$WEBKIT_FIX_DIR/libsoup-2.4.so.1"
ln -sf /usr/lib/x86_64-linux-gnu/libsoup-gnome-2.4.so.1 "$WEBKIT_FIX_DIR/libsoup-gnome-2.4.so.1"

# Also copy webkit if available
if [ -f "/usr/lib/x86_64-linux-gnu/libwebkit2gtk-4.0.so" ]; then
    ln -sf /usr/lib/x86_64-linux-gnu/libwebkit2gtk-4.0.so "$WEBKIT_FIX_DIR/libwebkit2gtk-4.0.so"
    ln -sf /usr/lib/x86_64-linux-gnu/libjavascriptcoregtk-4.0.so "$WEBKIT_FIX_DIR/libjavascriptcoregtk-4.0.so"
fi

# Create a launcher script
cat > "$WEBKIT_FIX_DIR/run-airimpute.sh" << 'EOF'
#!/bin/bash
# Run AirImpute with webkit fix

WEBKIT_FIX_DIR="$HOME/.airimpute-webkit-fix"
export LD_LIBRARY_PATH="$WEBKIT_FIX_DIR:$LD_LIBRARY_PATH"
export LD_PRELOAD="$WEBKIT_FIX_DIR/libsoup-2.4.so.1"

# Disable libsoup3 warning
export SOUP_DISABLE_COMPAT_CHECK=1

cd "$(dirname "$0")/../../"
npm run tauri dev
EOF

chmod +x "$WEBKIT_FIX_DIR/run-airimpute.sh"

echo ""
echo "WebKit fix environment created!"
echo "To run the application, use:"
echo "  $WEBKIT_FIX_DIR/run-airimpute.sh"
echo ""
echo "Or you can source this environment:"
echo "  export LD_LIBRARY_PATH=\"$WEBKIT_FIX_DIR:\$LD_LIBRARY_PATH\""
echo "  export LD_PRELOAD=\"$WEBKIT_FIX_DIR/libsoup-2.4.so.1\""
echo "  export SOUP_DISABLE_COMPAT_CHECK=1"