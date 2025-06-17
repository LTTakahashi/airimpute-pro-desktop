#!/bin/bash
# Combined launcher script for AirImpute Pro
# Implements Gemini's combined arms approach: build-time and run-time control

set -e

echo "==================================================================="
echo "AirImpute Pro - Combined libsoup2/3 Conflict Solution"
echo "==================================================================="
echo ""

# Find the absolute path to libsoup-2.4.so.1
LIBSOUP2_PATH=$(find /usr/lib /usr/lib64 /lib /lib64 -name "libsoup-2.4.so*" -print -quit 2>/dev/null | grep -v "\.0$" | head -1)

if [ -z "$LIBSOUP2_PATH" ]; then
    echo "Error: Could not find libsoup-2.4.so" >&2
    exit 1
fi

echo "Step 1: Build-time control"
echo "-------------------------"
echo "Using pkg-config wrapper to filter libsoup3 references..."

# Make pkg-config wrapper executable
chmod +x pkg-config-wrapper.sh

# Export our wrapper as the pkg-config to use
export PKG_CONFIG="$(pwd)/pkg-config-wrapper.sh"
echo "PKG_CONFIG=$PKG_CONFIG"

# Also set build flags to prefer our libraries
export RUSTFLAGS="-C link-arg=-Wl,--as-needed"

echo ""
echo "Step 2: Runtime control"
echo "----------------------"
echo "Preloading libsoup2 from: $LIBSOUP2_PATH"

# 1. LD_PRELOAD: Forces libsoup-2.4.so.1 to be loaded first
export LD_PRELOAD="$LIBSOUP2_PATH"

# 2. Disable webkit compositing to avoid potential issues
export WEBKIT_DISABLE_COMPOSITING_MODE=1

# 3. Force GI to use soup 2.4 (our Python fix handles this too)
export GI_TYPELIB_PATH="/usr/lib/x86_64-linux-gnu/girepository-1.0"

# 4. Set up webkit compatibility (use our existing script)
if [ -f "./tauri-dev.sh" ]; then
    # Source the webkit fix environment but don't execute
    WEBKIT_FIX_DIR="/tmp/webkit-fix-$$"
    rm -rf "$WEBKIT_FIX_DIR"
    mkdir -p "$WEBKIT_FIX_DIR"
    
    # Create the webkit 4.0 to 4.1 compatibility layer
    # Find the actual webkit 4.1 library
    WEBKIT_41=$(find /usr/lib* -name "libwebkit2gtk-4.1.so" -type l 2>/dev/null | head -1)
    if [ -n "$WEBKIT_41" ]; then
        ln -sf "$WEBKIT_41" "$WEBKIT_FIX_DIR/libwebkit2gtk-4.0.so"
    fi
    
    # Find javascriptcore 4.1
    JSCORE_41=$(find /usr/lib* -name "libjavascriptcoregtk-4.1.so" -type l 2>/dev/null | head -1)
    if [ -n "$JSCORE_41" ]; then
        ln -sf "$JSCORE_41" "$WEBKIT_FIX_DIR/libjavascriptcoregtk-4.0.so"
    fi
    
    # Create fixed .pc files that reference libsoup-2.4
    cat > "$WEBKIT_FIX_DIR/webkit2gtk-4.0.pc" << 'EOF'
prefix=/usr
exec_prefix=${prefix}
libdir=${prefix}/lib/x86_64-linux-gnu
includedir=${prefix}/include

Name: WebKit2GTK
Description: Web content engine for GTK applications
Version: 2.44.0
Requires: glib-2.0 >= 2.44.0, gobject-2.0 >= 2.44.0, gtk+-3.0 >= 3.22.0, libsoup-2.4 >= 2.54.0
Requires.private: gio-2.0 >= 2.44.0
Libs: -L${libdir} -lwebkit2gtk-4.1 -ljavascriptcoregtk-4.1
Cflags: -I${includedir}/webkitgtk-4.1
EOF

    cat > "$WEBKIT_FIX_DIR/javascriptcoregtk-4.0.pc" << 'EOF'
prefix=/usr
exec_prefix=${prefix}
libdir=${prefix}/lib/x86_64-linux-gnu
includedir=${prefix}/include

Name: JavaScriptCoreGTK
Description: JavaScript engine
Version: 2.44.0
Requires: glib-2.0 >= 2.44.0
Requires.private:
Libs: -L${libdir} -ljavascriptcoregtk-4.1
Cflags: -I${includedir}/webkitgtk-4.1
EOF

    export PKG_CONFIG_PATH="$WEBKIT_FIX_DIR:$PKG_CONFIG_PATH"
    export LD_LIBRARY_PATH="$WEBKIT_FIX_DIR:$LD_LIBRARY_PATH"
fi

echo ""
echo "Environment configured:"
echo "  LD_PRELOAD=$LD_PRELOAD"
echo "  PKG_CONFIG=$PKG_CONFIG"
echo "  PKG_CONFIG_PATH=$PKG_CONFIG_PATH"
echo ""

echo "Testing pkg-config output (should not contain -lsoup-3.0):"
echo "---------------------------------------------------------"
pkg-config --libs webkit2gtk-4.0 2>/dev/null | grep -o -- "-lsoup[^ ]*" || echo "✓ No soup libraries in output (good!)"
echo ""

echo "Starting application with combined fix..."
echo "========================================="
echo ""
echo "⚠️  WATCH FOR:"
echo "  - GType warnings about duplicate type registration"
echo "  - libsoup version conflict messages"
echo "  - Any crashes in webkit or soup functions"
echo ""

# Execute npm run tauri dev
exec npm run tauri dev "$@"