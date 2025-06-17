#!/bin/bash
# Build portable AirImpute Pro bundle without AppImage

set -e

echo "Building Portable AirImpute Pro Bundle..."
echo "========================================"

# Create bundle directory
BUNDLE_DIR="airimpute-pro-portable"
rm -rf "$BUNDLE_DIR"
mkdir -p "$BUNDLE_DIR/lib"

# Copy binary
cp target/release/airimpute-pro "$BUNDLE_DIR/"

# Copy Python wheels and scripts
mkdir -p "$BUNDLE_DIR/python"
cp -r scripts/airimpute "$BUNDLE_DIR/python/" 2>/dev/null || true

# Download compatible libraries
echo "Downloading compatible libraries..."
LIBS=(
    "http://archive.ubuntu.com/ubuntu/pool/main/libs/libsoup2.4/libsoup2.4-1_2.70.0-1_amd64.deb"
    "http://archive.ubuntu.com/ubuntu/pool/main/w/webkit2gtk/libwebkit2gtk-4.0-37_2.38.6-0ubuntu0.20.04.1_amd64.deb"
    "http://archive.ubuntu.com/ubuntu/pool/main/w/webkit2gtk/libjavascriptcoregtk-4.0-18_2.38.6-0ubuntu0.20.04.1_amd64.deb"
)

mkdir -p temp_extract
for lib_url in "${LIBS[@]}"; do
    lib_name=$(basename "$lib_url")
    echo "  Downloading $lib_name..."
    wget -q "$lib_url" -O "temp_extract/$lib_name"
    dpkg-deb -x "temp_extract/$lib_name" temp_extract/
done

# Copy only the necessary libraries
cp -P temp_extract/usr/lib/x86_64-linux-gnu/*.so* "$BUNDLE_DIR/lib/" 2>/dev/null || true

# Clean up
rm -rf temp_extract libsoup2.deb

# Create launcher script
cat > "$BUNDLE_DIR/airimpute-pro.sh" << 'EOF'
#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export LD_LIBRARY_PATH="$SCRIPT_DIR/lib:$LD_LIBRARY_PATH"
export WEBKIT_DISABLE_COMPOSITING_MODE=1
exec "$SCRIPT_DIR/airimpute-pro" "$@"
EOF

chmod +x "$BUNDLE_DIR/airimpute-pro.sh"

# Create README
cat > "$BUNDLE_DIR/README.txt" << 'EOF'
AirImpute Pro - Portable Linux Bundle
=====================================

This is a portable bundle that includes compatible libraries
to work around the libsoup2/libsoup3 conflict on modern Linux systems.

To run:
  ./airimpute-pro.sh

This bundle includes:
- AirImpute Pro binary
- libsoup 2.4 (compatible version)
- webkit2gtk 4.0 libraries

Note: This is a temporary solution. Future versions will use
Tauri v2 which properly handles these library conflicts.
EOF

# Create tarball
echo "Creating portable bundle..."
tar -czf airimpute-pro-portable-linux-x64.tar.gz "$BUNDLE_DIR"

echo ""
echo "Portable bundle created: airimpute-pro-portable-linux-x64.tar.gz"
echo "Extract and run with: ./airimpute-pro.sh"
echo ""