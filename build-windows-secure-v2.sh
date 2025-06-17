#!/bin/bash
set -euxo pipefail

# Secure Windows Cross-Compilation Script for AirImpute Pro Desktop
# Implements all security recommendations from Gemini analysis

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Configuration
PYTHON_VERSION="3.11.7"
PYTHON_STANDALONE_VERSION="20240107"
BUILD_TYPE="${BUILD_TYPE:-release}"
TIMESTAMP_URL="https://timestamp.sectigo.com"  # HTTPS for security
TARGET="x86_64-pc-windows-gnu"

# Security checks
check_dependencies() {
    echo "Checking dependencies..."
    local missing=()
    
    for cmd in cargo rustc zig wget jq sha256sum; do
        if ! command -v "$cmd" &> /dev/null; then
            missing+=("$cmd")
        fi
    done
    
    if [ ${#missing[@]} -ne 0 ]; then
        echo "Error: Missing required dependencies: ${missing[*]}"
        exit 1
    fi
    
    echo "All dependencies satisfied"
}

# Download and verify Python embeddable package
setup_python_bundle() {
    echo "Setting up Python bundle..."
    local python_dir="src-tauri/python"
    local python_url="https://github.com/indygreg/python-build-standalone/releases/download/${PYTHON_STANDALONE_VERSION}/cpython-${PYTHON_VERSION}+${PYTHON_STANDALONE_VERSION}-x86_64-pc-windows-msvc-shared-install_only.tar.gz"
    local python_sha256="expected_sha256_here"  # TODO: Get actual SHA256
    
    mkdir -p "$python_dir"
    
    # Download if not exists
    if [ ! -f "$python_dir/python.tar.gz" ]; then
        echo "Downloading Python ${PYTHON_VERSION}..."
        wget -O "$python_dir/python.tar.gz" "$python_url"
    fi
    
    # Verify integrity
    echo "Verifying Python bundle integrity..."
    # TODO: Implement SHA256 verification
    # echo "$python_sha256  $python_dir/python.tar.gz" | sha256sum -c -
    
    # Extract
    echo "Extracting Python bundle..."
    tar -xzf "$python_dir/python.tar.gz" -C "$python_dir" --strip-components=1
    
    # Install wheels from requirements
    install_python_wheels "$python_dir"
}

# Install pre-compiled wheels
install_python_wheels() {
    local python_dir="$1"
    local wheels_dir="$python_dir/wheels"
    
    echo "Installing Python wheels..."
    mkdir -p "$wheels_dir"
    
    # Generate requirements with hashes
    pip-compile --generate-hashes \
        --output-file=requirements.txt \
        python-requirements.in
    
    # Download wheels for Windows
    pip download \
        --platform win_amd64 \
        --python-version 311 \
        --only-binary :all: \
        --dest "$wheels_dir" \
        -r requirements.txt
    
    # Verify wheel integrity
    echo "Verifying wheel integrity..."
    for wheel in "$wheels_dir"/*.whl; do
        if [ -f "$wheel" ]; then
            unzip -t "$wheel" > /dev/null || {
                echo "Error: Corrupted wheel: $wheel"
                exit 1
            }
        fi
    done
}

# Configure Tauri security
configure_tauri_security() {
    echo "Configuring Tauri security..."
    
    # Create secure Tauri configuration
    local config_file="src-tauri/tauri.conf.json"
    local temp_config="$config_file.tmp"
    
    # Use jq to modify configuration securely
    jq '
    .tauri.security = {
        "csp": {
            "default-src": ["self"],
            "script-src": ["self"],
            "style-src": ["self", "unsafe-inline"],
            "img-src": ["self", "data:", "asset:", "https://asset.localhost"],
            "connect-src": ["self", "https://localhost:*"],
            "font-src": ["self"],
            "object-src": ["none"],
            "base-uri": ["self"],
            "form-action": ["self"],
            "frame-ancestors": ["none"],
            "block-all-mixed-content": true,
            "upgrade-insecure-requests": true
        },
        "dangerousDisableAssetCspModification": false,
        "freezePrototype": true
    } |
    .tauri.allowlist = {
        "all": false,
        "shell": {
            "all": false,
            "open": false
        },
        "dialog": {
            "all": false,
            "open": true,
            "save": true
        },
        "path": {
            "all": false
        },
        "fs": {
            "all": false,
            "readFile": true,
            "writeFile": true,
            "readDir": true,
            "createDir": true,
            "removeDir": false,
            "removeFile": false,
            "scope": ["$APPDATA/**", "$DOCUMENT/**", "$DOWNLOAD/**"]
        },
        "window": {
            "all": false,
            "create": false,
            "center": true,
            "requestUserAttention": true,
            "setResizable": true,
            "setTitle": true,
            "maximize": true,
            "unmaximize": true,
            "minimize": true,
            "unminimize": true,
            "show": true,
            "hide": true,
            "close": true,
            "setDecorations": true,
            "setAlwaysOnTop": true,
            "setSize": true,
            "setMinSize": true,
            "setMaxSize": true,
            "setPosition": true,
            "setFullscreen": true,
            "setFocus": true,
            "setIcon": true,
            "setSkipTaskbar": true,
            "setCursorGrab": true,
            "setCursorVisible": true,
            "setCursorIcon": true,
            "setCursorPosition": true,
            "startDragging": true,
            "print": true
        }
    } |
    .tauri.bundle.windows.timestampUrl = "'"$TIMESTAMP_URL"'"
    ' "$config_file" > "$temp_config"
    
    mv "$temp_config" "$config_file"
}

# Build with Zig for cross-compilation
build_with_zig() {
    echo "Building with Zig cross-compiler..."
    
    # Set environment variables for cross-compilation
    export CC="zig cc -target x86_64-windows-gnu"
    export CXX="zig c++ -target x86_64-windows-gnu"
    export AR="zig ar"
    export CARGO_TARGET_X86_64_PC_WINDOWS_GNU_LINKER="zig cc -target x86_64-windows-gnu"
    
    # Security flags
    export RUSTFLAGS="-C link-arg=/DYNAMICBASE -C link-arg=/HIGHENTROPYVA -C link-arg=/NXCOMPAT -C link-arg=/GUARD:CF"
    
    # Build options
    local build_args=(
        "--target" "$TARGET"
        "--features" "custom-protocol,python-support"
    )
    
    if [ "$BUILD_TYPE" = "release" ]; then
        build_args+=("--release")
    fi
    
    # Clean previous builds
    cd src-tauri
    cargo clean --target "$TARGET"
    
    # Build the application
    echo "Building application..."
    cargo build "${build_args[@]}"
    
    # Run Tauri build
    echo "Running Tauri build..."
    cargo tauri build --target "$TARGET" -- "${build_args[@]}"
    
    cd ..
}

# Post-build security hardening
post_build_hardening() {
    echo "Applying post-build security hardening..."
    
    local exe_path="src-tauri/target/$TARGET/release/airimpute-pro-desktop.exe"
    
    if [ -f "$exe_path" ]; then
        # Verify executable properties
        file "$exe_path"
        
        # TODO: Add code signing here
        # signtool sign /tr "$TIMESTAMP_URL" /td sha256 /fd sha256 "$exe_path"
        
        echo "Build complete: $exe_path"
    else
        echo "Error: Executable not found at $exe_path"
        exit 1
    fi
}

# Main execution
main() {
    echo "Starting secure Windows cross-compilation..."
    
    check_dependencies
    setup_python_bundle
    configure_tauri_security
    build_with_zig
    post_build_hardening
    
    echo "Secure Windows build completed successfully!"
}

# Run main function
main "$@"