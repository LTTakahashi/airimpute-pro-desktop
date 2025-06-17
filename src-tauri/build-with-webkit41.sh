#!/bin/bash
# Build script that creates symlinks from webkit 4.0 to 4.1

# Create a temporary directory for our library symlinks
LINK_DIR="/tmp/webkit-symlinks-$$"
mkdir -p "$LINK_DIR"

# Find the actual library locations
WEBKIT_41=$(find /usr/lib -name "libwebkit2gtk-4.1.so*" -type f | head -1)
JSCORE_41=$(find /usr/lib -name "libjavascriptcoregtk-4.1.so*" -type f | head -1)

if [ -z "$WEBKIT_41" ] || [ -z "$JSCORE_41" ]; then
    echo "Error: Could not find webkit2gtk-4.1 libraries"
    echo "Please install: sudo apt install libwebkit2gtk-4.1-dev"
    exit 1
fi

echo "Found webkit 4.1 at: $WEBKIT_41"
echo "Found javascriptcore 4.1 at: $JSCORE_41"

# Create symlinks with 4.0 names pointing to 4.1 libraries
ln -sf "$WEBKIT_41" "$LINK_DIR/libwebkit2gtk-4.0.so"
ln -sf "$JSCORE_41" "$LINK_DIR/libjavascriptcoregtk-4.0.so"

# Also create versioned symlinks
ln -sf "$WEBKIT_41" "$LINK_DIR/libwebkit2gtk-4.0.so.37"
ln -sf "$JSCORE_41" "$LINK_DIR/libjavascriptcoregtk-4.0.so.18"

# Export the library path
export LD_LIBRARY_PATH="$LINK_DIR:$LD_LIBRARY_PATH"
export LIBRARY_PATH="$LINK_DIR:$LIBRARY_PATH"

# Set Python configuration for PyO3
export PYO3_PYTHON=/usr/bin/python3

# Add Python library path
PYTHON_LIB_DIR=$(python3-config --prefix)/lib
export LD_LIBRARY_PATH="$PYTHON_LIB_DIR:$LD_LIBRARY_PATH"
export LIBRARY_PATH="$PYTHON_LIB_DIR:$LIBRARY_PATH"

echo "Building with webkit 4.1 symlinked as 4.0..."

# Run the build
cargo build "$@"
BUILD_RESULT=$?

# Clean up
rm -rf "$LINK_DIR"

exit $BUILD_RESULT