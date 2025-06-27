#!/bin/bash

# Fix Python build error on WSL for Tauri application
# This script sets up proper cross-compilation for Windows target on WSL

echo "=== Fixing Python Build Error on WSL ==="
echo

# Set working directory
cd "$(dirname "$0")"

# Check if we're on WSL
if ! grep -qi microsoft /proc/version; then
    echo "Warning: This script is designed for WSL. Proceeding anyway..."
fi

echo "Detected issue: Building for Windows target on WSL with Windows Python distribution"
echo

# Option 1: Use system Python for build (recommended for WSL)
echo "Option 1: Using system Python for the build process"
echo "This is the recommended approach for building on WSL"
echo

# Check for Python 3.11 on the system
if command -v python3.11 &> /dev/null; then
    PYTHON_CMD="python3.11"
    echo "Found Python 3.11 on system"
elif command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
    PYTHON_VERSION=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1-2)
    echo "Found Python $PYTHON_VERSION on system"
else
    echo "Error: No Python found on system. Please install Python 3.11:"
    echo "  sudo apt update && sudo apt install python3.11 python3.11-dev"
    exit 1
fi

# Set environment variables for native build
export PYO3_PYTHON="$PYTHON_CMD"
unset PYO3_CROSS_LIB_DIR
unset PYO3_CROSS_PYTHON_VERSION
unset PYO3_CROSS_PYTHON_IMPLEMENTATION
unset PYO3_CROSS

echo "Environment variables set for native build:"
echo "  PYO3_PYTHON=$PYO3_PYTHON"
echo

# Option 2: Create a proper cross-compilation setup
echo "Option 2: Set up proper cross-compilation (advanced)"
echo "For cross-compiling to Windows from WSL, you need:"
echo

# Create a minimal _sysconfigdata for Windows Python
cat << 'EOF' > src-tauri/python/_sysconfigdata__win32_.py
# Minimal sysconfigdata for Windows Python 3.11 cross-compilation
build_time_vars = {
    'ABIFLAGS': '',
    'BINDIR': '',
    'BINLIBDEST': '',
    'EXT_SUFFIX': '.pyd',
    'INCLUDEPY': '',
    'LIBDEST': '',
    'LIBDIR': '',
    'LIBPC': '',
    'LDLIBRARY': 'python311.dll',
    'LIBRARY': 'python311.lib',
    'PREFIX': '',
    'SOABI': 'cp311-win_amd64',
    'VERSION': '3.11',
    'Py_DEBUG': 0,
    'Py_ENABLE_SHARED': 1,
    'SIZEOF_VOID_P': 8,
}
EOF

echo "Created minimal _sysconfigdata for Windows Python"

# Build script for WSL
cat << 'EOF' > build-wsl.sh
#!/bin/bash

echo "Building Tauri app on WSL..."

# Clean previous builds
cd src-tauri
cargo clean
cd ..

# For native Linux build (testing on WSL)
if [ "$1" == "linux" ]; then
    echo "Building for Linux target..."
    npm run tauri build
    
# For Windows cross-compilation
elif [ "$1" == "windows" ]; then
    echo "Building for Windows target..."
    
    # Install Windows target if not present
    rustup target add x86_64-pc-windows-gnu
    
    # Set up cross-compilation environment
    export PYO3_CROSS_LIB_DIR="$(pwd)/src-tauri/python"
    export PYO3_CROSS_PYTHON_VERSION="3.11"
    export PYO3_CROSS_PYTHON_IMPLEMENTATION="CPython"
    export PYO3_CROSS="1"
    export CARGO_TARGET_X86_64_PC_WINDOWS_GNU_LINKER="x86_64-w64-mingw32-gcc"
    
    # Build for Windows
    cd src-tauri
    cargo build --release --target x86_64-pc-windows-gnu
    cd ..
    
else
    echo "Usage: ./build-wsl.sh [linux|windows]"
    echo "  linux   - Build native Linux version (for testing on WSL)"
    echo "  windows - Cross-compile for Windows"
fi
EOF

chmod +x build-wsl.sh

echo
echo "=== Fix Applied ==="
echo
echo "To build your application on WSL, you have several options:"
echo
echo "1. Build native Linux version (for testing):"
echo "   ./build-wsl.sh linux"
echo
echo "2. Install Python 3.11 development files and build:"
echo "   sudo apt update"
echo "   sudo apt install python3.11 python3.11-dev"
echo "   npm run tauri build"
echo
echo "3. For Windows cross-compilation (advanced):"
echo "   sudo apt install mingw-w64"
echo "   ./build-wsl.sh windows"
echo
echo "4. Alternative: Build on native Windows instead of WSL"
echo "   This avoids cross-compilation complexity"
echo
echo "Note: The bundled Windows Python in src-tauri/python/ is meant for"
echo "      the final Windows executable, not for building on Linux/WSL."