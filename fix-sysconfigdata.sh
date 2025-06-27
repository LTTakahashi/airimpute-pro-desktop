#!/bin/bash

# Quick fix to create a minimal _sysconfigdata file for Windows Python on WSL

PYTHON_DIR="src-tauri/python/Lib"

echo "Creating minimal _sysconfigdata for Windows Python bundled in project..."

# Create a minimal _sysconfigdata file
cat > "$PYTHON_DIR/_sysconfigdata__win32_.py" << 'EOF'
# Minimal sysconfigdata for Windows Python on WSL build
# This allows pyo3-ffi to find required configuration

build_time_vars = {
    'EXT_SUFFIX': '.pyd',
    'SHLIB_SUFFIX': '.pyd',
    'SO': '.pyd',
    'SOABI': 'cp311-win_amd64',
    'Py_ENABLE_SHARED': 1,
    'LIBDIR': '',
    'BINDIR': '',
    'INCLUDEPY': '',
    'VERSION': '3.11',
    'prefix': '',
    'exec_prefix': '',
}
EOF

echo "Created $PYTHON_DIR/_sysconfigdata__win32_.py"

# Now try building with the bundled Python
echo "Attempting build with bundled Python..."
cd src-tauri || exit 1

# Set environment to use bundled Python
export PYO3_CROSS_LIB_DIR="$(pwd)/python"
export PYO3_CROSS_PYTHON_VERSION="3.11"

cargo build --release