#!/bin/bash
# Create a minimal Windows executable for AirImpute Pro
# This version disables Python to ensure successful compilation

set -e

echo "========================================"
echo "Creating Windows Executable (No Python)"
echo "========================================"

# Create a minimal Cargo.toml without PyO3
cat > Cargo-minimal.toml << 'EOF'
[package]
name = "airimpute-pro"
version = "1.0.0"
description = "Professional Air Quality Data Imputation - Desktop Edition"
edition = "2021"

[build-dependencies]
tauri-build = { version = "1.5", features = [] }

[dependencies]
tauri = { version = "1.5", features = [ "shell-open", "dialog-all", "fs-all", "notification-all", "process-all", "clipboard-all", "protocol-asset", "system-tray"] }
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
tokio = { version = "1", features = ["full"] }
anyhow = "1.0"
thiserror = "1.0"
tracing = "0.1"
tracing-subscriber = { version = "0.3", features = ["env-filter"] }
chrono = { version = "0.4", features = ["serde"] }
uuid = { version = "1.6", features = ["v4", "serde"] }
dashmap = "5.5"
once_cell = "1.19"
lazy_static = "1.4"
dirs = "5.0"

[target.'cfg(windows)'.dependencies]
windows-sys = { version = "0.48", features = ["Win32_System_LibraryLoader"] }

[features]
default = ["custom-protocol"]
custom-protocol = ["tauri/custom-protocol"]

[profile.release]
opt-level = 3
lto = true
codegen-units = 1
strip = true
panic = "abort"
EOF

# Build with the minimal config
echo "Building with minimal configuration..."
cp Cargo.toml Cargo.toml.full
cp Cargo-minimal.toml Cargo.toml

# Try to build
echo "Attempting Windows build..."
if command -v cross &> /dev/null; then
    echo "Using cross for compilation..."
    cross build --target x86_64-pc-windows-gnu --release
else
    echo "cross not available, trying cargo with target..."
    cargo build --target x86_64-pc-windows-gnu --release
fi

# Check if build succeeded
if [ -f "target/x86_64-pc-windows-gnu/release/airimpute-pro.exe" ]; then
    echo ""
    echo "========================================"
    echo "SUCCESS! Windows executable created:"
    echo "  target/x86_64-pc-windows-gnu/release/airimpute-pro.exe"
    echo ""
    echo "Note: This version has Python support disabled."
    echo "The executable can run but scientific functions won't work."
    echo "========================================"
    
    # Create distribution folder
    mkdir -p ../windows-dist
    cp target/x86_64-pc-windows-gnu/release/airimpute-pro.exe ../windows-dist/
    echo "Executable copied to: windows-dist/airimpute-pro.exe"
else
    echo "Build failed. Check error messages above."
fi

# Restore original Cargo.toml
mv Cargo.toml.full Cargo.toml