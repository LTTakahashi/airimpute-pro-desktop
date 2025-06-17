#!/bin/bash
# Script to run AirImpute Pro with various compatibility options

echo "AirImpute Pro Desktop - Development Runner"
echo "========================================="
echo ""
echo "Detected libsoup2/libsoup3 conflict on modern Linux systems."
echo "Choose a solution to run the application:"
echo ""
echo "1) Nix Shell (cleanest isolation) - Requires Nix installation"
echo "2) Bundled Libraries (download compatible libs) - No extra tools needed" 
echo "3) Docker Container (full isolation) - Requires Docker"
echo "4) Native with webkit workaround (may still fail)"
echo "5) Native without workaround (will fail on modern systems)"
echo ""
read -p "Enter your choice (1-5): " choice

case $choice in
    1)
        echo "Running with Nix shell..."
        if ! command -v nix-shell &> /dev/null; then
            echo "Nix is not installed. Would you like to install it? (y/n)"
            read -p "> " install_nix
            if [ "$install_nix" = "y" ]; then
                echo "Please run: sh <(curl -L https://nixos.org/nix/install) --daemon"
                echo "Then restart your terminal and run this script again."
            fi
            exit 1
        fi
        ./run-with-nix.sh
        ;;
    2)
        echo "Running with bundled libraries..."
        ./run-bundled.sh
        ;;
    3)
        echo "Running in Docker container..."
        if ! command -v docker &> /dev/null; then
            echo "Docker is not installed. Please install Docker first."
            exit 1
        fi
        ./docker-dev.sh
        ;;
    4)
        echo "Running natively with webkit workaround..."
        ./tauri-dev.sh
        ;;
    5)
        echo "Running natively without workaround..."
        echo "WARNING: This will likely fail with libsoup conflict!"
        npm run tauri dev
        ;;
    *)
        echo "Invalid choice. Exiting."
        exit 1
        ;;
esac