#!/bin/bash
# Quick diagnostic script using release binary

set -e

echo "==================================================================="
echo "AirImpute Pro - Quick libsoup Conflict Diagnostic"
echo "==================================================================="
echo ""

# Use release binary
BINARY="src-tauri/target/release/airimpute-pro"
if [ ! -f "$BINARY" ]; then
    echo "Error: Release binary not found at $BINARY"
    echo "Please build first with: npm run build"
    exit 1
fi

# Create diagnostic output directory
DIAG_DIR="diagnostics-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$DIAG_DIR"

echo "Diagnostic output will be saved to: $DIAG_DIR/"
echo ""

# Method 1: LD_DEBUG - The Gold Standard
echo "Step 1: Running with LD_DEBUG to trace library loading..."
echo "------------------------------------------------------"
export LD_DEBUG=libs
export LD_DEBUG_OUTPUT="$DIAG_DIR/ld_debug"

# Run the app and capture output
timeout 3s "$BINARY" 2>&1 | tee "$DIAG_DIR/runtime_output.log" || true

# Find libsoup loading
echo ""
echo "Step 2: Analyzing libsoup loading..."
echo "-----------------------------------"
if ls "$DIAG_DIR"/ld_debug.* 1> /dev/null 2>&1; then
    # Get the first libsoup3 mention and its context
    grep -B10 -A5 "libsoup.*3" "$DIAG_DIR"/ld_debug.* | head -50 > "$DIAG_DIR/libsoup3_context.txt" || true
    
    # Find what requested libsoup
    grep -B5 "libsoup" "$DIAG_DIR"/ld_debug.* | grep -E "(calling init:|needed by)" > "$DIAG_DIR/soup_requesters.txt" || true
    
    echo "Key findings:"
    echo "-------------"
    if [ -s "$DIAG_DIR/libsoup3_context.txt" ]; then
        echo "✗ Found libsoup3 loading attempt!"
        echo ""
        echo "Context around first libsoup3 load:"
        grep -E "(calling init:|needed by|trying file.*libsoup)" "$DIAG_DIR/libsoup3_context.txt" | head -10
    else
        echo "✓ No libsoup3 loading detected"
    fi
fi

# Method 2: Check Python modules quickly
echo ""
echo "Step 3: Checking Python environment..."
echo "-------------------------------------"
python3 -c "
import sys
print(f'Python: {sys.version}')
print('Checking for gi module...')
try:
    import gi
    print('✗ gi (GObject Introspection) is installed - PRIMARY SUSPECT')
    try:
        # Check what gi might load
        gi.require_version('Soup', '2.4')
        print('  → Successfully requested Soup 2.4')
    except:
        print('  → Failed to request Soup 2.4 (might default to 3.0)')
except ImportError:
    print('✓ gi module not installed')
" > "$DIAG_DIR/python_check.txt" 2>&1

cat "$DIAG_DIR/python_check.txt"

# Generate quick summary
echo ""
echo "Step 4: Summary"
echo "--------------"
{
    echo "Quick Diagnostic Summary"
    echo "======================="
    echo ""
    
    if grep -q "libsoup.*3" "$DIAG_DIR/runtime_output.log" 2>/dev/null; then
        echo "CONFLICT CONFIRMED: libsoup3 symbols detected"
        echo ""
        grep "libsoup" "$DIAG_DIR/runtime_output.log" | head -5
    fi
    
    echo ""
    echo "MOST LIKELY CAUSE:"
    if [ -s "$DIAG_DIR/soup_requesters.txt" ]; then
        echo "Libraries that requested libsoup:"
        cat "$DIAG_DIR/soup_requesters.txt" | head -10
    fi
    
    echo ""
    echo "Check $DIAG_DIR/ for detailed analysis"
} | tee "$DIAG_DIR/SUMMARY.txt"

echo ""
echo "==================================================================="
echo "Diagnostic complete! Results in: $DIAG_DIR/"
echo "==================================================================="