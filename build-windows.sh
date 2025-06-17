#!/bin/bash
# Build script for Windows cross-compilation
# Implements security-hardened build based on Gemini's analysis

set -e

echo "========================================"
echo "AirImpute Pro - Windows Build Script"
echo "========================================"
echo ""

# Check if we're running on Windows (Git Bash/WSL) or Linux
if [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "win32" ]]; then
    echo "Running on Windows - using native build"
    NATIVE_BUILD=true
else
    echo "Running on Linux - setting up cross-compilation"
    NATIVE_BUILD=false
fi

# Step 1: Install cross-compilation tools if on Linux
if [ "$NATIVE_BUILD" = false ]; then
    echo "Step 1: Installing Windows cross-compilation tools..."
    
    # Check if cross is installed
    if ! command -v cross &> /dev/null; then
        echo "Installing 'cross' for cross-compilation..."
        cargo install cross --git https://github.com/cross-rs/cross
    fi
    
    # Check if mingw is installed for Windows builds
    if ! command -v x86_64-w64-mingw32-gcc &> /dev/null; then
        echo "ERROR: MinGW-w64 is required for cross-compilation."
        echo "Please install it with: sudo apt-get install mingw-w64"
        exit 1
    fi
fi

# Step 2: Prepare Windows-specific configuration
echo ""
echo "Step 2: Preparing Windows-specific configuration..."

# Copy Windows-specific Cargo.toml
cp src-tauri/Cargo-windows.toml src-tauri/Cargo.toml.bak
cp src-tauri/Cargo-windows.toml src-tauri/Cargo.toml

# Copy Windows-specific cargo config
cp .cargo/config-windows.toml .cargo/config.toml.bak 2>/dev/null || true
cp .cargo/config-windows.toml .cargo/config.toml

# Step 3: Update tauri.conf.json for Windows
echo ""
echo "Step 3: Configuring Tauri for Windows build..."

# Create a Windows-specific tauri config
cat > src-tauri/tauri-windows.conf.json << 'EOF'
{
  "$schema": "../node_modules/@tauri-apps/cli/schema.json",
  "build": {
    "beforeDevCommand": "npm run dev",
    "beforeBuildCommand": "npm run build",
    "devPath": "http://localhost:5173",
    "distDir": "../dist",
    "withGlobalTauri": false
  },
  "package": {
    "productName": "AirImpute Pro",
    "version": "1.0.0"
  },
  "tauri": {
    "allowlist": {
      "all": false,
      "shell": {
        "all": false,
        "open": true
      },
      "fs": {
        "all": false,
        "readFile": true,
        "writeFile": true,
        "readDir": true,
        "createDir": true,
        "removeDir": true,
        "removeFile": true,
        "renameFile": true,
        "exists": true,
        "scope": ["$APPDATA/**", "$LOCALAPPDATA/**", "$DOCUMENT/AirImpute/**", "$TEMP/**"]
      },
      "path": {
        "all": false
      },
      "dialog": {
        "all": false,
        "open": true,
        "save": true,
        "message": true,
        "ask": true,
        "confirm": true
      },
      "clipboard": {
        "all": false,
        "writeText": true,
        "readText": true
      },
      "notification": {
        "all": true
      },
      "globalShortcut": {
        "all": false
      },
      "os": {
        "all": false
      },
      "process": {
        "all": false,
        "relaunch": true,
        "exit": true
      },
      "protocol": {
        "all": false,
        "asset": true,
        "assetScope": ["**"]
      }
    },
    "bundle": {
      "active": true,
      "targets": ["msi", "nsis"],
      "identifier": "com.airimpute.pro",
      "publisher": "AirImpute Team",
      "copyright": "© 2024 AirImpute Research Team",
      "category": "Science",
      "shortDescription": "Professional Air Quality Data Imputation",
      "longDescription": "AirImpute Pro is a professional desktop application for air quality data imputation using scientifically validated methods including RAH, statistical, and machine learning approaches.",
      "icon": [
        "icons/32x32.png",
        "icons/128x128.png",
        "icons/128x128@2x.png",
        "icons/icon.ico"
      ],
      "resources": [
        "migrations/*"
      ],
      "windows": {
        "certificateThumbprint": null,
        "digestAlgorithm": "sha256",
        "timestampUrl": "http://timestamp.digicert.com",
        "wix": {
          "language": "en-US"
        },
        "nsis": {
          "installerIcon": "icons/icon.ico",
          "installMode": "perUser",
          "languages": ["English"],
          "displayLanguageSelector": false
        },
        "webviewInstallMode": {
          "type": "downloadBootstrapper"
        }
      }
    },
    "security": {
      "csp": "default-src 'self'; img-src 'self' asset: https://asset.localhost blob: data:; style-src 'self' 'unsafe-inline'; script-src 'self'; connect-src 'self' ipc: https://ipc.localhost; frame-src 'none'; object-src 'none'; base-uri 'self'; form-action 'self'; font-src 'self' asset: https://asset.localhost"
    },
    "systemTray": {
      "iconPath": "icons/icon.ico",
      "iconAsTemplate": false,
      "menuOnLeftClick": false
    },
    "updater": {
      "active": false
    },
    "windows": [
      {
        "title": "AirImpute Pro",
        "label": "main",
        "width": 1400,
        "height": 900,
        "minWidth": 1024,
        "minHeight": 600,
        "resizable": true,
        "fullscreen": false,
        "center": true,
        "decorations": true,
        "transparent": false,
        "skipTaskbar": false,
        "alwaysOnTop": false,
        "visible": true,
        "fileDropEnabled": true,
        "theme": "Dark"
      }
    ]
  }
}
EOF

# Step 4: Build frontend
echo ""
echo "Step 4: Building frontend..."
npm run build

# Step 5: Build the Windows executable
echo ""
echo "Step 5: Building Windows executable..."
cd src-tauri

if [ "$NATIVE_BUILD" = true ]; then
    # Native Windows build
    cargo build --release --target x86_64-pc-windows-msvc
else
    # Cross-compilation from Linux
    echo "Cross-compiling for Windows..."
    
    # Use cross for building
    cross build --release --target x86_64-pc-windows-gnu
fi

# Step 6: Create Windows distribution
echo ""
echo "Step 6: Creating Windows distribution package..."

DIST_DIR="../dist-windows"
rm -rf "$DIST_DIR"
mkdir -p "$DIST_DIR"

# Copy the executable
if [ "$NATIVE_BUILD" = true ]; then
    cp target/x86_64-pc-windows-msvc/release/airimpute-pro.exe "$DIST_DIR/"
else
    cp target/x86_64-pc-windows-gnu/release/airimpute-pro.exe "$DIST_DIR/"
fi

# Copy necessary files
cp -r ../dist/* "$DIST_DIR/" 2>/dev/null || true

# Create a simple batch file launcher
cat > "$DIST_DIR/AirImpute Pro.bat" << 'EOF'
@echo off
start "" "%~dp0airimpute-pro.exe"
EOF

# Step 7: Restore original configuration
echo ""
echo "Step 7: Cleaning up..."
cd ..
mv src-tauri/Cargo.toml.bak src-tauri/Cargo.toml
mv .cargo/config.toml.bak .cargo/config.toml 2>/dev/null || true

echo ""
echo "========================================"
echo "Windows build complete!"
echo ""
echo "Output location: dist-windows/"
echo "Executable: dist-windows/airimpute-pro.exe"
echo ""
echo "IMPORTANT NOTES:"
echo "1. The executable is unsigned - it will trigger Windows Defender warnings"
echo "2. For production, obtain a code signing certificate"
echo "3. Python functionality requires bundling Python (not included in this build)"
echo "4. HDF5/NetCDF support is disabled in this Windows build"
echo ""
echo "To test the build, copy dist-windows/ to a Windows machine and run."
echo "========================================"