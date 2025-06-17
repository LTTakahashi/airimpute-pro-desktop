#!/bin/bash
# Clean build and run script for AirImpute Pro

echo "🧹 Cleaning build artifacts..."
cd src-tauri
cargo clean
cd ..

echo "🚀 Starting fresh build with webkit compatibility..."
./tauri-dev.sh "$@"