#!/bin/bash
# Build wrapper for Tauri app with webkit2gtk-4.1

# Create temporary directory for symlinks
TEMP_LIB_DIR=$(mktemp -d)
echo "Creating temporary library directory: $TEMP_LIB_DIR"

# Find the actual webkit libraries
WEBKIT_41=$(pkg-config --variable=libdir webkit2gtk-4.1)/libwebkit2gtk-4.1.so
JSCORE_41=$(pkg-config --variable=libdir webkit2gtk-4.1)/libjavascriptcoregtk-4.1.so

# Create symlinks
ln -sf "$WEBKIT_41" "$TEMP_LIB_DIR/libwebkit2gtk-4.0.so"
ln -sf "$JSCORE_41" "$TEMP_LIB_DIR/libjavascriptcoregtk-4.0.so"

echo "Created symlinks:"
ls -la "$TEMP_LIB_DIR/"

# Set library path
export LD_LIBRARY_PATH="$TEMP_LIB_DIR:$LD_LIBRARY_PATH"
export LIBRARY_PATH="$TEMP_LIB_DIR:$LIBRARY_PATH"

# Change to src-tauri directory
cd src-tauri

# Build based on argument
if [ "$1" = "dev" ]; then
    echo "Building in development mode..."
    cargo build
else
    echo "Building in release mode..."
    cargo build --release
fi

BUILD_STATUS=$?

# Cleanup
rm -rf "$TEMP_LIB_DIR"
echo "Cleaned up temporary directory"

exit $BUILD_STATUS