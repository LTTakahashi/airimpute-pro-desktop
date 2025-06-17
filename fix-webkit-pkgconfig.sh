#!/bin/bash
# Advanced fix for webkit pkg-config to remove libsoup3 references
# Based on Gemini's root cause analysis

set -e

echo "==================================================================="
echo "AirImpute Pro - Advanced webkit pkg-config fix"
echo "==================================================================="
echo ""

# Create a temporary directory for our fixed pkg-config files
WEBKIT_FIX_DIR="/tmp/webkit-fix-advanced-$$"
rm -rf "$WEBKIT_FIX_DIR"
mkdir -p "$WEBKIT_FIX_DIR"

echo "Creating fixed pkg-config files in: $WEBKIT_FIX_DIR"

# Create a wrapper script for pkg-config that filters out libsoup3
cat > "$WEBKIT_FIX_DIR/pkg-config" << 'EOF'
#!/bin/bash
# Wrapper for pkg-config that removes libsoup3 references

# Call the real pkg-config
REAL_PKGCONFIG="/usr/bin/pkg-config"
OUTPUT=$($REAL_PKGCONFIG "$@" 2>&1)
EXIT_CODE=$?

# If this is a webkit2gtk query, filter out libsoup3
if [[ "$*" =~ webkit2gtk ]] && [[ "$OUTPUT" =~ libsoup ]]; then
    # Replace libsoup-3.0 references with libsoup-2.4
    OUTPUT=$(echo "$OUTPUT" | sed 's/-lsoup-3\.0/-lsoup-2.4/g')
    OUTPUT=$(echo "$OUTPUT" | sed 's|/include/libsoup-3\.0|/include/libsoup-2.4|g')
    
    # Remove any libsoup3 include paths
    OUTPUT=$(echo "$OUTPUT" | sed 's|-I/usr/include/libsoup-3\.0||g')
fi

echo "$OUTPUT"
exit $EXIT_CODE
EOF

chmod +x "$WEBKIT_FIX_DIR/pkg-config"

# Create fixed .pc files
echo "Creating fixed webkit2gtk-4.0.pc..."
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
Conflicts:
Libs: -L${libdir} -lwebkit2gtk-4.1 -ljavascriptcoregtk-4.1
Libs.private:
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
Conflicts:
Libs: -L${libdir} -ljavascriptcoregtk-4.1
Libs.private:
Cflags: -I${includedir}/webkitgtk-4.1
EOF

# Create symlinks for library compatibility
ln -sf /usr/lib/x86_64-linux-gnu/libwebkit2gtk-4.1.so "$WEBKIT_FIX_DIR/libwebkit2gtk-4.0.so"
ln -sf /usr/lib/x86_64-linux-gnu/libjavascriptcoregtk-4.1.so "$WEBKIT_FIX_DIR/libjavascriptcoregtk-4.0.so"

# Export environment variables
export PATH="$WEBKIT_FIX_DIR:$PATH"
export PKG_CONFIG_PATH="$WEBKIT_FIX_DIR:$PKG_CONFIG_PATH"
export LD_LIBRARY_PATH="$WEBKIT_FIX_DIR:$LD_LIBRARY_PATH"

# Also force libsoup2 preload just in case
export LD_PRELOAD="/usr/lib/x86_64-linux-gnu/libsoup-2.4.so.1.11.2"

# Disable compositing
export WEBKIT_DISABLE_COMPOSITING_MODE=1

echo ""
echo "Environment configured:"
echo "PKG_CONFIG_PATH=$PKG_CONFIG_PATH"
echo "PATH=$PATH"
echo "LD_PRELOAD=$LD_PRELOAD"
echo ""

# Test the fix
echo "Testing pkg-config output:"
echo "-------------------------"
pkg-config --libs webkit2gtk-4.0 2>/dev/null || echo "webkit2gtk-4.0 query failed"
echo ""

# Make sure Rust uses our fixed environment
export RUSTFLAGS="-L$WEBKIT_FIX_DIR"

echo "Starting Tauri with fixed configuration..."
echo ""

# Run tauri dev
exec npm run tauri dev "$@"