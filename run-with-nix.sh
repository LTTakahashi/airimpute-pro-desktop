#!/bin/bash
# Run AirImpute Pro using Nix shell for webkit compatibility

set -e

echo "Starting AirImpute Pro with Nix environment..."
echo "=========================================="

# Check if nix is installed
if ! command -v nix-shell &> /dev/null; then
    echo "Error: Nix is not installed!"
    echo ""
    echo "Please install Nix first:"
    echo "  sh <(curl -L https://nixos.org/nix/install) --daemon"
    echo ""
    echo "After installation, restart your terminal and run this script again."
    exit 1
fi

# Check if npm dependencies are installed
if [ ! -d "node_modules" ]; then
    echo "Installing npm dependencies..."
    npm install
fi

# Enter nix shell and run the app
echo "Entering Nix shell environment..."
nix-shell --run "npm run tauri dev"