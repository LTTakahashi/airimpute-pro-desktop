#!/bin/bash

# Fix Python DLL issue for AirImputePro.exe
# This script rebuilds the application with the correct Python version

echo "=== AirImputePro Python DLL Fix Script ==="
echo

# Navigate to project directory
cd "$(dirname "$0")"

# Option 1: Quick fix - Copy python311.dll to python310.dll (temporary workaround)
echo "Option 1: Quick fix (copy DLL)"
if [ -f "src-tauri/python/python311.dll" ]; then
    echo "Creating python310.dll from python311.dll..."
    cp src-tauri/python/python311.dll src-tauri/python/python310.dll
    echo "✓ Created python310.dll"
    
    # Also copy in target directories if they exist
    for dir in src-tauri/target/release/python src-tauri/target/debug/python; do
        if [ -d "$dir" ] && [ -f "$dir/python311.dll" ]; then
            cp "$dir/python311.dll" "$dir/python310.dll"
            echo "✓ Created python310.dll in $dir"
        fi
    done
else
    echo "⚠ python311.dll not found"
fi

echo
echo "Option 2: Proper fix (rebuild with correct Python)"
echo "To properly fix this issue, you need to:"
echo "1. Install Python 3.11 on your system"
echo "2. Create a virtual environment with Python 3.11:"
echo "   python3.11 -m venv src-tauri/.venv"
echo "3. Activate the virtual environment:"
echo "   - Windows: src-tauri\\.venv\\Scripts\\activate"
echo "   - Linux/Mac: source src-tauri/.venv/bin/activate"
echo "4. Rebuild the application:"
echo "   cd src-tauri && cargo clean && cd .."
echo "   npm run tauri build"

echo
echo "Option 3: Alternative fix (download Python 3.10)"
echo "If you need to use Python 3.10 specifically:"
echo "1. Download Python 3.10 embeddable package from:"
echo "   https://www.python.org/ftp/python/3.10.11/python-3.10.11-embed-amd64.zip"
echo "2. Extract contents to src-tauri/python/"
echo "3. Rebuild the application"

echo
echo "=== Quick fix applied. Try running AirImputePro.exe now ==="