#!/bin/bash
# Debug script to trace libsoup loading

set -e

echo "Debugging libsoup loading..."
echo "==========================="

# Build debug binary if needed
if [ ! -f "src-tauri/target/debug/airimpute-pro" ]; then
    echo "Building debug binary..."
    cd src-tauri
    cargo build
    cd ..
fi

# Method 1: Use LD_DEBUG to trace library loading
echo "Starting with LD_DEBUG trace..."
export LD_DEBUG=libs
export LD_DEBUG_OUTPUT=/tmp/ld-debug

# Run the binary directly to see what loads
timeout 5s src-tauri/target/debug/airimpute-pro 2>&1 | tee /tmp/airimpute-debug.log || true

# Analyze the debug output
echo ""
echo "Analyzing library loading..."
if [ -f "/tmp/ld-debug.*" ]; then
    echo "Libraries that tried to load libsoup:"
    grep -E "libsoup|calling init:|needed by" /tmp/ld-debug.* | grep -B2 -A2 "libsoup" | head -50
fi

# Method 2: Use ldd on the binary
echo ""
echo "Static dependencies of the binary:"
ldd src-tauri/target/debug/airimpute-pro | grep -E "soup|webkit"

# Method 3: Check what Python modules might be loading
echo ""
echo "Checking Python modules..."
python3 << 'EOF'
import sys
print(f"Python: {sys.version}")
print("\nChecking for modules that might use libsoup:")

# Common culprits
suspicious_modules = [
    'gi', 'gi.repository', 'keyring', 'secretstorage', 
    'jeepney', 'dbus', 'notify2', 'plyer'
]

for module in suspicious_modules:
    try:
        __import__(module)
        print(f"  ✗ {module} is installed (potential libsoup user)")
    except ImportError:
        print(f"  ✓ {module} is not installed")
EOF

# Method 4: Try to run with explicit libsoup2 preload
echo ""
echo "Attempting to force libsoup2..."
if [ -f "/usr/lib/x86_64-linux-gnu/libsoup-2.4.so.1" ]; then
    export LD_PRELOAD="/usr/lib/x86_64-linux-gnu/libsoup-2.4.so.1"
    echo "LD_PRELOAD set to libsoup2"
else
    echo "libsoup2 not found on system"
fi

# Clean up
rm -f /tmp/ld-debug.* /tmp/airimpute-debug.log