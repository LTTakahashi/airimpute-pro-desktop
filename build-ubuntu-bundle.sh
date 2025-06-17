#!/bin/bash
# Build script for creating a bundled AppImage with libsoup2 compatibility

set -e

echo "Building AirImpute Pro with bundled libraries..."
echo "=============================================="

# Check if we're on Ubuntu 22.04 or compatible
if ! grep -q "Ubuntu 22.04" /etc/os-release 2>/dev/null; then
    echo "Warning: This script is designed for Ubuntu 22.04"
    echo "Your system may have different library versions"
fi

# Install build dependencies
echo "Installing build dependencies..."
sudo apt-get update
sudo apt-get install -y \
    build-essential \
    curl \
    wget \
    file \
    libssl-dev \
    libgtk-3-dev \
    libayatana-appindicator3-dev \
    librsvg2-dev \
    libwebkit2gtk-4.0-dev \
    libsoup2.4-dev \
    python3-dev \
    python3-pip \
    pkg-config \
    libglib2.0-dev \
    libcairo2-dev \
    libpango1.0-dev \
    libgdk-pixbuf2.0-dev

# Build the application
echo "Building application..."
npm install
npm run build

# Build Tauri with release profile
echo "Building Tauri release..."
cd src-tauri
cargo build --release
cd ..

# Create AppImage structure
echo "Creating AppImage structure..."
APP_DIR="airimpute-pro.AppDir"
rm -rf "$APP_DIR"
mkdir -p "$APP_DIR/usr/bin"
mkdir -p "$APP_DIR/usr/lib"
mkdir -p "$APP_DIR/usr/share/applications"
mkdir -p "$APP_DIR/usr/share/icons/hicolor/256x256/apps"

# Copy binary
cp src-tauri/target/release/airimpute-pro "$APP_DIR/usr/bin/"

# Copy icon
cp src-tauri/icons/icon.png "$APP_DIR/usr/share/icons/hicolor/256x256/apps/airimpute-pro.png"

# Create desktop file
cat > "$APP_DIR/usr/share/applications/airimpute-pro.desktop" << EOF
[Desktop Entry]
Name=AirImpute Pro
Exec=airimpute-pro
Icon=airimpute-pro
Type=Application
Categories=Science;Education;
Comment=Professional Air Quality Data Imputation
EOF

# Copy required libraries
echo "Copying required libraries..."
# This function copies a library and its dependencies
copy_deps() {
    local lib=$1
    if [ -f "$lib" ]; then
        cp -L "$lib" "$APP_DIR/usr/lib/" 2>/dev/null || true
        ldd "$lib" 2>/dev/null | grep "=> /" | awk '{print $3}' | while read dep; do
            if [ -f "$dep" ] && [[ "$dep" == *"soup"* || "$dep" == *"webkit"* || "$dep" == *"javascriptcore"* ]]; then
                cp -L "$dep" "$APP_DIR/usr/lib/" 2>/dev/null || true
            fi
        done
    fi
}

# Copy webkit and soup libraries
copy_deps /usr/lib/x86_64-linux-gnu/libwebkit2gtk-4.0.so.37
copy_deps /usr/lib/x86_64-linux-gnu/libjavascriptcoregtk-4.0.so.18
copy_deps /usr/lib/x86_64-linux-gnu/libsoup-2.4.so.1
copy_deps /usr/lib/x86_64-linux-gnu/libsoup-gnome-2.4.so.1

# Create AppRun script
cat > "$APP_DIR/AppRun" << 'EOF'
#!/bin/bash
SELF=$(readlink -f "$0")
HERE=${SELF%/*}
export LD_LIBRARY_PATH="${HERE}/usr/lib:${LD_LIBRARY_PATH}"
export WEBKIT_DISABLE_COMPOSITING_MODE=1
exec "${HERE}/usr/bin/airimpute-pro" "$@"
EOF
chmod +x "$APP_DIR/AppRun"

# Download appimagetool if not present
if [ ! -f "appimagetool-x86_64.AppImage" ]; then
    echo "Downloading appimagetool..."
    wget -q https://github.com/AppImage/AppImageKit/releases/download/continuous/appimagetool-x86_64.AppImage
    chmod +x appimagetool-x86_64.AppImage
fi

# Create AppImage
echo "Creating AppImage..."
./appimagetool-x86_64.AppImage "$APP_DIR" "AirImpute-Pro-x86_64.AppImage"

echo ""
echo "Build complete!"
echo "AppImage created: AirImpute-Pro-x86_64.AppImage"
echo ""
echo "This AppImage bundles libsoup2 and webkit2gtk-4.0 for compatibility."
echo "It should work on any modern Linux system without library conflicts."