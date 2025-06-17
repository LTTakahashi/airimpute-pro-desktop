#!/bin/bash
# Run AirImpute Pro in development mode with bundled libraries

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP_DIR="$(dirname "$SCRIPT_DIR")"

echo "Starting AirImpute Pro with bundled libraries..."
echo "=============================================="

# Set up library path with our bundled libraries
export LD_LIBRARY_PATH="$SCRIPT_DIR/lib:$LD_LIBRARY_PATH"
export WEBKIT_DISABLE_COMPOSITING_MODE=1

# Change to app directory
cd "$APP_DIR"

# Check if frontend dev server is running
if ! curl -s http://localhost:5173 > /dev/null 2>&1; then
    echo "Starting frontend development server..."
    npm run dev &
    FRONTEND_PID=$!
    
    # Wait for frontend to start
    echo "Waiting for frontend to start..."
    for i in {1..30}; do
        if curl -s http://localhost:5173 > /dev/null 2>&1; then
            echo "Frontend server started!"
            break
        fi
        sleep 1
    done
fi

# Run the app with bundled libraries
echo "Starting Tauri app..."
cd src-tauri
export LD_LIBRARY_PATH="../airimpute-pro-portable/lib:$LD_LIBRARY_PATH"
cargo tauri dev

# Cleanup
if [ ! -z "$FRONTEND_PID" ]; then
    kill $FRONTEND_PID 2>/dev/null
fi