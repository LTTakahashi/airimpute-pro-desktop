#!/bin/bash
# Development script for AirImpute Pro with all necessary checks

echo "🚀 Starting AirImpute Pro Development Server..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if we're in the right directory
if [ ! -f "package.json" ] || [ ! -d "src-tauri" ]; then
    echo -e "${RED}Error: Must run from airimpute-pro-desktop directory${NC}"
    exit 1
fi

# Check Node.js
echo -e "${YELLOW}Checking Node.js...${NC}"
if ! command -v node &> /dev/null; then
    echo -e "${RED}Node.js is not installed${NC}"
    exit 1
fi
node_version=$(node --version)
echo -e "${GREEN}✓ Node.js ${node_version}${NC}"

# Check npm
echo -e "${YELLOW}Checking npm...${NC}"
if ! command -v npm &> /dev/null; then
    echo -e "${RED}npm is not installed${NC}"
    exit 1
fi
npm_version=$(npm --version)
echo -e "${GREEN}✓ npm ${npm_version}${NC}"

# Check Rust
echo -e "${YELLOW}Checking Rust...${NC}"
if ! command -v cargo &> /dev/null; then
    echo -e "${RED}Rust is not installed${NC}"
    echo "Please install Rust from https://rustup.rs/"
    exit 1
fi
rust_version=$(rustc --version)
echo -e "${GREEN}✓ ${rust_version}${NC}"

# Check Python
echo -e "${YELLOW}Checking Python...${NC}"
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}Python 3 is not installed${NC}"
    exit 1
fi
python_version=$(python3 --version)
echo -e "${GREEN}✓ ${python_version}${NC}"

# Check if node_modules exists
if [ ! -d "node_modules" ]; then
    echo -e "${YELLOW}Installing npm dependencies...${NC}"
    npm install
fi

# Check WebKit libraries
echo -e "${YELLOW}Checking WebKit libraries...${NC}"
if ! pkg-config --exists webkit2gtk-4.1 2>/dev/null; then
    echo -e "${YELLOW}WebKit 4.1 found - using compatibility mode${NC}"
else
    echo -e "${GREEN}✓ WebKit 4.1 available${NC}"
fi

# Kill any existing process on port 5173
if lsof -i :5173 > /dev/null 2>&1; then
    echo -e "${YELLOW}Killing existing process on port 5173...${NC}"
    kill $(lsof -t -i:5173) 2>/dev/null || true
    sleep 1
fi

# Use the tauri-dev.sh script which handles webkit compatibility
echo -e "${GREEN}Starting development server...${NC}"
echo -e "${YELLOW}Note: Python integration will use subprocess mode${NC}"
echo -e "${YELLOW}For full Python integration, ensure required packages are installed${NC}"
echo ""

# Make sure the script is executable
chmod +x tauri-dev.sh

# Alternative: Run with environment variables directly
echo -e "${YELLOW}Setting up webkit compatibility layer...${NC}"

# Create temporary directory for webkit compatibility
WEBKIT_FIX_DIR="/tmp/webkit-fix-$$"
mkdir -p "$WEBKIT_FIX_DIR"

# Create pkg-config files
cat > "$WEBKIT_FIX_DIR/webkit2gtk-4.0.pc" << 'EOF'
Name: WebKit2GTK
Description: Web content engine
Version: 2.48.1
Requires: webkit2gtk-4.1
Libs: -lwebkit2gtk-4.1
Cflags:
EOF

cat > "$WEBKIT_FIX_DIR/javascriptcoregtk-4.0.pc" << 'EOF'
Name: JavaScriptCore
Description: JavaScript engine
Version: 2.48.1
Requires: javascriptcoregtk-4.1
Libs: -ljavascriptcoregtk-4.1
Cflags:
EOF

# Export environment
export PKG_CONFIG_PATH="$WEBKIT_FIX_DIR:$PKG_CONFIG_PATH"
export PYO3_PYTHON=/usr/bin/python3
export RUSTFLAGS="-C link-arg=-lpython3.12"

# Cleanup function
cleanup() {
    echo -e "${YELLOW}Cleaning up...${NC}"
    rm -rf "$WEBKIT_FIX_DIR"
}
trap cleanup EXIT

# Run tauri dev directly
echo -e "${GREEN}Starting Tauri development server...${NC}"
npm run tauri dev "$@"