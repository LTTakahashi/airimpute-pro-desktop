#!/bin/bash
# Fix Python linking issue by creating a custom pkg-config file

echo "Creating custom Python pkg-config file..."

# Create a custom directory for our fixed pkg-config files
mkdir -p ~/.config/pkgconfig

# Create a fixed python3.pc file
cat > ~/.config/pkgconfig/python3.pc << 'EOF'
# Fixed pkg-config file for Python 3.12
prefix=/usr
exec_prefix=${prefix}
includedir=${prefix}/include
libdir=${exec_prefix}/lib/x86_64-linux-gnu

Name: Python
Description: Build a C extension for Python
Requires:
Version: 3.12
Libs: -L${libdir} -lpython3.12
Libs.private: -ldl -lm
Cflags: -I${includedir}/python3.12 -I${includedir}/x86_64-linux-gnu/python3.12
EOF

# Set PKG_CONFIG_PATH to use our custom file
export PKG_CONFIG_PATH=~/.config/pkgconfig:$PKG_CONFIG_PATH

echo "Testing pkg-config..."
pkg-config --libs python3

echo "Building with fixed Python linking..."
cargo build