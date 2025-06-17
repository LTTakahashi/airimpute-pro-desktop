#!/bin/bash
# Fix libsoup conflict using rpath approach
# Based on Gemini's Strategy B recommendation

set -e

echo "==================================================================="
echo "AirImpute Pro - libsoup Conflict Fix (rpath method)"
echo "==================================================================="
echo ""

# Check for required tools
if ! command -v patchelf &> /dev/null; then
    echo "Installing patchelf..."
    sudo apt-get update && sudo apt-get install -y patchelf
fi

# Create bundled libs directory
BUNDLE_DIR="bundled-libs"
rm -rf "$BUNDLE_DIR"
mkdir -p "$BUNDLE_DIR"

echo "Step 1: Identifying required libraries..."
echo "----------------------------------------"

# Find the exact libsoup-2.4 library
LIBSOUP2=$(find /usr/lib* -name "libsoup-2.4.so*" -type f 2>/dev/null | grep -v "\.0$" | head -1)
if [ -z "$LIBSOUP2" ]; then
    echo "Error: libsoup-2.4 not found on system"
    echo "Please install: sudo apt-get install libsoup2.4-1"
    exit 1
fi
echo "Found libsoup2: $LIBSOUP2"

# Find webkit2gtk-4.0 libraries
WEBKIT=$(find /usr/lib* -name "libwebkit2gtk-4.0.so*" -type f 2>/dev/null | head -1)
JAVASCRIPTCORE=$(find /usr/lib* -name "libjavascriptcoregtk-4.0.so*" -type f 2>/dev/null | head -1)

if [ -z "$WEBKIT" ] || [ -z "$JAVASCRIPTCORE" ]; then
    echo "Warning: webkit2gtk-4.0 libraries not found"
    echo "The application may still work if webkit 4.1 is compatible"
fi

echo ""
echo "Step 2: Copying required libraries..."
echo "------------------------------------"

# Copy libsoup2 and its direct dependencies
copy_library_deps() {
    local lib=$1
    local libname=$(basename "$lib")
    
    if [ -f "$lib" ] && [ ! -f "$BUNDLE_DIR/$libname" ]; then
        echo "Copying: $libname"
        cp -L "$lib" "$BUNDLE_DIR/" 2>/dev/null || true
        
        # Get dependencies but filter out system libraries
        ldd "$lib" 2>/dev/null | grep "=> /" | awk '{print $3}' | while read dep; do
            local depname=$(basename "$dep")
            # Only copy soup-related dependencies and specific GTK/GLib libraries
            if [[ "$depname" =~ ^lib(soup|glib|gobject|gio|gmodule|gtk|gdk|pango|cairo|atk|harfbuzz|fontconfig|freetype|pixbuf|rsvg|webkit|javascriptcore).*\.so ]]; then
                if [ ! -f "$BUNDLE_DIR/$depname" ]; then
                    echo "  → Dependency: $depname"
                    cp -L "$dep" "$BUNDLE_DIR/" 2>/dev/null || true
                fi
            fi
        done
    fi
}

# Copy critical libraries
copy_library_deps "$LIBSOUP2"

# Also copy libsoup-gnome if it exists
LIBSOUP_GNOME=$(find /usr/lib* -name "libsoup-gnome-2.4.so.1" -type f 2>/dev/null | head -1)
if [ -n "$LIBSOUP_GNOME" ]; then
    copy_library_deps "$LIBSOUP_GNOME"
fi

# Copy webkit libraries if found
if [ -n "$WEBKIT" ]; then
    echo "Copying webkit2gtk-4.0..."
    copy_library_deps "$WEBKIT"
fi

if [ -n "$JAVASCRIPTCORE" ]; then
    echo "Copying javascriptcoregtk-4.0..."
    copy_library_deps "$JAVASCRIPTCORE"
fi

echo ""
echo "Step 3: Patching the binary with rpath..."
echo "----------------------------------------"

# Find the binary
BINARY="src-tauri/target/debug/airimpute-pro"
BINARY_RELEASE="src-tauri/target/release/airimpute-pro"

if [ -f "$BINARY_RELEASE" ]; then
    BINARY="$BINARY_RELEASE"
    echo "Using release binary: $BINARY"
elif [ -f "$BINARY" ]; then
    echo "Using debug binary: $BINARY"
else
    echo "Error: No binary found. Please build first with: npm run build"
    exit 1
fi

# Get the directory containing the binary
BINARY_DIR=$(dirname "$(readlink -f "$BINARY")")

# Copy bundled libs next to the binary
FINAL_BUNDLE_DIR="$BINARY_DIR/bundled-libs"
echo "Copying bundled libraries to: $FINAL_BUNDLE_DIR"
rm -rf "$FINAL_BUNDLE_DIR"
cp -r "$BUNDLE_DIR" "$FINAL_BUNDLE_DIR"

# Patch the binary to use bundled libs
echo "Patching binary rpath..."
patchelf --set-rpath '$ORIGIN/bundled-libs:$ORIGIN/../lib:$ORIGIN' "$BINARY"

# Verify the patch
echo ""
echo "Step 4: Verifying..."
echo "-------------------"
echo "New rpath:"
patchelf --print-rpath "$BINARY"

echo ""
echo "Libraries in bundle:"
ls -la "$FINAL_BUNDLE_DIR" | grep "\.so" | wc -l
echo ""

# Create a wrapper script
WRAPPER="run-airimpute-fixed.sh"
cat > "$WRAPPER" << 'EOF'
#!/bin/bash
# Wrapper to run AirImpute Pro with fixed libraries

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Ensure our bundled libs are used first
export LD_LIBRARY_PATH="$SCRIPT_DIR/src-tauri/target/release/bundled-libs:$SCRIPT_DIR/src-tauri/target/debug/bundled-libs:$LD_LIBRARY_PATH"

# Disable compositing mode which can cause issues
export WEBKIT_DISABLE_COMPOSITING_MODE=1

# Force libsoup2 if needed
export GI_TYPELIB_PATH="/usr/lib/x86_64-linux-gnu/girepository-1.0"

echo "Running AirImpute Pro with bundled libraries..."
echo "LD_LIBRARY_PATH=$LD_LIBRARY_PATH"
echo ""

# Run the dev server
exec npm run tauri dev "$@"
EOF

chmod +x "$WRAPPER"

echo "==================================================================="
echo "Fix applied successfully!"
echo "==================================================================="
echo ""
echo "The binary has been patched to use bundled libsoup2 libraries."
echo ""
echo "To run the application with the fix:"
echo "  ./$WRAPPER"
echo ""
echo "Or directly:"
echo "  npm run tauri dev"
echo ""
echo "The bundled libraries are in: $FINAL_BUNDLE_DIR"
echo "This should prevent the libsoup2/libsoup3 conflict."
echo ""