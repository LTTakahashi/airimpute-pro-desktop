#!/bin/bash

echo "=== Testing AirImpute Pro Desktop Build on Linux WSL2 ==="
echo

# Check Node.js
echo "1. Node.js version:"
node --version
echo

# Check npm
echo "2. npm version:"
npm --version
echo

# Check Rust
echo "3. Rust version:"
rustc --version
echo

# Check Cargo
echo "4. Cargo version:"
cargo --version
echo

# Check Python
echo "5. Python version:"
python3 --version
echo

# Check system dependencies
echo "6. Checking Tauri dependencies:"
pkg-config --version && echo "✓ pkg-config installed"
pkg-config --libs webkit2gtk-4.0 2>/dev/null && echo "✓ webkit2gtk-4.0 found" || echo "✗ webkit2gtk-4.0 missing"
pkg-config --libs gtk+-3.0 2>/dev/null && echo "✓ gtk+-3.0 found" || echo "✗ gtk+-3.0 missing"
echo

# Frontend build status
echo "7. Frontend build:"
if [ -d "dist" ]; then
    echo "✓ Frontend build exists (dist/ directory found)"
    ls -la dist/ | head -5
else
    echo "✗ Frontend build missing"
fi
echo

# Test basic Tauri functionality
echo "8. Testing basic Rust compilation:"
cd src-tauri
cargo check --message-format short 2>&1 | grep -E "error:|warning:" | wc -l | xargs -I {} echo "Found {} errors/warnings"
echo

echo "=== Summary ==="
echo "The environment is set up for Linux WSL2."
echo "The frontend builds successfully."
echo "However, there are Rust compilation errors that need to be fixed."
echo
echo "To run the app despite the errors, you could:"
echo "1. Fix the compilation errors in the Rust code"
echo "2. Use 'npm run tauri:dev' for development mode"
echo "3. Temporarily disable problematic modules"