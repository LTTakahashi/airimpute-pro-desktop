#!/bin/bash
# Create a temporary directory for our webkit 4.0 -> 4.1 redirects
WEBKIT_FIX_DIR="/tmp/webkit-fix-$$"
mkdir -p "$WEBKIT_FIX_DIR"

# Create pkg-config files that redirect to 4.1
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

# Export the path so pkg-config finds our redirect files first
export PKG_CONFIG_PATH="$WEBKIT_FIX_DIR:$PKG_CONFIG_PATH"

# Run the build
cargo build "$@"

# Clean up
rm -rf "$WEBKIT_FIX_DIR"