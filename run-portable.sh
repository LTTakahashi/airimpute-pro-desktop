#!/bin/bash
# Run AirImpute Pro with the portable bundle libraries

echo "Starting AirImpute Pro with portable libraries..."
echo "==============================================="

# Set up environment with bundled libraries
export LD_LIBRARY_PATH="$(pwd)/airimpute-pro-portable/lib:$LD_LIBRARY_PATH"
export WEBKIT_DISABLE_COMPOSITING_MODE=1

echo "Using bundled libraries from: $(pwd)/airimpute-pro-portable/lib"
echo ""

# Run using npm/tauri dev which handles both frontend and backend
npm run tauri dev