#!/bin/bash
# Build a fully bundled AirImpute Pro application with all dependencies
# This creates a relocatable bundle that works on any Linux system

set -e

echo "Building bundled AirImpute Pro application..."
echo "==========================================="

# Check for required tools
if ! command -v docker &> /dev/null; then
    echo "Error: Docker is required for this build process"
    echo "Please install Docker or use the manual build method"
    exit 1
fi

# Create output directory
OUTPUT_DIR="dist-bundled"
rm -rf "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR"

# Create a temporary Dockerfile for building
cat > Dockerfile.bundle << 'EOF'
FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    wget \
    file \
    patchelf \
    libssl-dev \
    libgtk-3-dev \
    libayatana-appindicator3-dev \
    librsvg2-dev \
    libwebkit2gtk-4.0-dev \
    libsoup2.4-dev \
    python3.11 \
    python3.11-dev \
    python3-pip \
    pkg-config \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Rust
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

# Install Node.js
RUN curl -fsSL https://deb.nodesource.com/setup_18.x | bash - && \
    apt-get install -y nodejs

WORKDIR /app
EOF

# Build the Docker image
echo "Building Docker image..."
docker build -f Dockerfile.bundle -t airimpute-builder .

# Run the build inside Docker
echo "Running build in Docker container..."
docker run --rm \
    -v "$(pwd):/app" \
    -v "$(pwd)/$OUTPUT_DIR:/output" \
    airimpute-builder \
    bash -c '
set -e

# Install dependencies
echo "Installing npm dependencies..."
npm install

# Build frontend
echo "Building frontend..."
npm run build

# Build Rust backend
echo "Building Rust backend..."
cd src-tauri
cargo build --release
cd ..

# Create bundle directory structure
BUNDLE_DIR="/output/airimpute-pro"
rm -rf "$BUNDLE_DIR"
mkdir -p "$BUNDLE_DIR/bin"
mkdir -p "$BUNDLE_DIR/lib"
mkdir -p "$BUNDLE_DIR/share/icons"

# Copy the binary
cp src-tauri/target/release/airimpute-pro "$BUNDLE_DIR/bin/"

# Copy icon
cp src-tauri/icons/icon.png "$BUNDLE_DIR/share/icons/airimpute-pro.png"

# Function to copy library and its dependencies
copy_lib_deps() {
    local lib=$1
    local dest=$2
    
    if [ -f "$lib" ]; then
        cp -L "$lib" "$dest/" 2>/dev/null || true
        
        # Get dependencies
        ldd "$lib" 2>/dev/null | grep "=> /" | awk "{print \$3}" | while read dep; do
            # Skip system libraries we dont need to bundle
            if [[ ! "$dep" =~ libc\.so|libm\.so|libdl\.so|libpthread\.so|libresolv\.so|librt\.so|ld-linux ]]; then
                if [ -f "$dep" ]; then
                    cp -L "$dep" "$dest/" 2>/dev/null || true
                fi
            fi
        done
    fi
}

# Copy all required libraries
echo "Copying required libraries..."

# Get all dependencies of our binary
ldd src-tauri/target/release/airimpute-pro | grep "=> /" | awk "{print \$3}" | while read lib; do
    # Only copy non-system libraries
    if [[ "$lib" =~ webkit|soup|javascript|gtk|gdk|cairo|pango|atk|gio|glib|gobject|pixbuf|rsvg|python ]]; then
        copy_lib_deps "$lib" "$BUNDLE_DIR/lib"
    fi
done

# Specifically ensure we have the right webkit and soup versions
copy_lib_deps /usr/lib/x86_64-linux-gnu/libwebkit2gtk-4.0.so.37 "$BUNDLE_DIR/lib"
copy_lib_deps /usr/lib/x86_64-linux-gnu/libjavascriptcoregtk-4.0.so.18 "$BUNDLE_DIR/lib"
copy_lib_deps /usr/lib/x86_64-linux-gnu/libsoup-2.4.so.1 "$BUNDLE_DIR/lib"
copy_lib_deps /usr/lib/x86_64-linux-gnu/libsoup-gnome-2.4.so.1 "$BUNDLE_DIR/lib"

# Patch the binary to use our bundled libraries
echo "Patching binary..."
patchelf --set-rpath "\$ORIGIN/../lib" "$BUNDLE_DIR/bin/airimpute-pro"

# Create launcher script
cat > "$BUNDLE_DIR/airimpute-pro.sh" << "LAUNCHER"
#!/bin/bash
# Launcher for AirImpute Pro

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export LD_LIBRARY_PATH="$SCRIPT_DIR/lib:$LD_LIBRARY_PATH"
export WEBKIT_DISABLE_COMPOSITING_MODE=1

# Ensure Python is available
export PYTHONHOME=""
export PYTHONPATH=""

exec "$SCRIPT_DIR/bin/airimpute-pro" "$@"
LAUNCHER

chmod +x "$BUNDLE_DIR/airimpute-pro.sh"

# Create desktop file
cat > "$BUNDLE_DIR/airimpute-pro.desktop" << "DESKTOP"
[Desktop Entry]
Name=AirImpute Pro
Exec=%k/airimpute-pro.sh
Icon=%k/share/icons/airimpute-pro.png
Type=Application
Categories=Science;Education;
Comment=Professional Air Quality Data Imputation
DESKTOP

echo "Bundle created successfully!"
'

# Clean up
rm -f Dockerfile.bundle

# Create a tar.gz archive
echo "Creating archive..."
cd "$OUTPUT_DIR"
tar -czf airimpute-pro-linux-x64.tar.gz airimpute-pro/
cd ..

echo ""
echo "Build complete!"
echo "=============="
echo "Bundle location: $OUTPUT_DIR/airimpute-pro/"
echo "Archive: $OUTPUT_DIR/airimpute-pro-linux-x64.tar.gz"
echo ""
echo "To run the application:"
echo "  cd $OUTPUT_DIR/airimpute-pro"
echo "  ./airimpute-pro.sh"
echo ""
echo "This bundle includes all required libraries and should work on any"
echo "modern Linux system without dependency conflicts."