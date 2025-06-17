#!/bin/bash
# Comprehensive diagnostic script to identify the source of libsoup3 loading
# Based on Gemini's gold standard approach

set -e

echo "==================================================================="
echo "AirImpute Pro - libsoup Conflict Diagnostic Tool"
echo "==================================================================="
echo ""

# Ensure we have a debug binary
if [ ! -f "src-tauri/target/debug/airimpute-pro" ]; then
    echo "Building debug binary for analysis..."
    cd src-tauri
    cargo build
    cd ..
fi

# Create diagnostic output directory
DIAG_DIR="diagnostics-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$DIAG_DIR"

echo "Diagnostic output will be saved to: $DIAG_DIR/"
echo ""

# Method 1: LD_DEBUG - The Gold Standard
echo "Step 1: Running with LD_DEBUG to trace library loading..."
echo "------------------------------------------------------"
export LD_DEBUG=libs,symbols,files
export LD_DEBUG_OUTPUT="$DIAG_DIR/ld_debug"

# Run the app and capture output
timeout 5s src-tauri/target/debug/airimpute-pro 2>&1 | tee "$DIAG_DIR/runtime_output.log" || true

# Analyze the debug output
echo ""
echo "Step 2: Analyzing library loading sequence..."
echo "--------------------------------------------"
if ls "$DIAG_DIR"/ld_debug.* 1> /dev/null 2>&1; then
    # Find the first mention of libsoup3
    echo "Looking for first libsoup3 load..."
    grep -n "libsoup.*3" "$DIAG_DIR"/ld_debug.* | head -20 > "$DIAG_DIR/libsoup3_first_load.txt" || true
    
    # Get context around libsoup loads
    echo "Getting context around libsoup loads..."
    grep -B5 -A5 "libsoup" "$DIAG_DIR"/ld_debug.* | grep -E "(calling init:|needed by|find library=)" > "$DIAG_DIR/libsoup_context.txt" || true
    
    # Find who's calling what
    echo "Identifying library dependencies..."
    grep -E "(calling init:|needed by)" "$DIAG_DIR"/ld_debug.* | grep -B2 -A2 "soup" > "$DIAG_DIR/soup_callers.txt" || true
fi

# Method 2: Static dependency analysis
echo ""
echo "Step 3: Static dependency analysis..."
echo "-----------------------------------"
echo "Direct dependencies of airimpute-pro:" > "$DIAG_DIR/static_deps.txt"
ldd src-tauri/target/debug/airimpute-pro >> "$DIAG_DIR/static_deps.txt" 2>&1

echo "" >> "$DIAG_DIR/static_deps.txt"
echo "Webkit and soup dependencies:" >> "$DIAG_DIR/static_deps.txt"
ldd src-tauri/target/debug/airimpute-pro | grep -E "(webkit|soup)" >> "$DIAG_DIR/static_deps.txt" 2>&1 || true

# Method 3: Python module analysis
echo ""
echo "Step 4: Checking Python modules that might load libsoup3..."
echo "---------------------------------------------------------"
python3 << 'EOF' > "$DIAG_DIR/python_modules.txt" 2>&1
import sys
import subprocess
import importlib.util

print(f"Python version: {sys.version}")
print(f"Python executable: {sys.executable}")
print("\nChecking for modules that might use libsoup:")
print("=" * 50)

# List of modules that commonly use libsoup via gi
suspicious_modules = [
    ('gi', 'GObject Introspection - Primary suspect'),
    ('gi.repository', 'GObject Introspection repository'),
    ('keyring', 'System keyring access (may use Secret service)'),
    ('secretstorage', 'Secret Service client (uses D-Bus)'),
    ('jeepney', 'Pure Python D-Bus library'),
    ('dbus', 'D-Bus Python bindings'),
    ('notify2', 'Desktop notifications'),
    ('plyer', 'Platform-specific features'),
    ('requests', 'HTTP library (check for system integration)'),
    ('urllib3', 'HTTP library (check for system integration)'),
]

for module_name, description in suspicious_modules:
    try:
        spec = importlib.util.find_spec(module_name)
        if spec is not None:
            print(f"\n✗ {module_name} is installed")
            print(f"  Description: {description}")
            print(f"  Location: {spec.origin if spec.origin else 'built-in'}")
            
            # Try to check if it's using libsoup
            if module_name == 'gi':
                try:
                    import gi
                    print("  Checking gi versions available:")
                    # This is safe - we're just checking, not loading
                    result = subprocess.run(['python3', '-c', 
                        "import gi; print('    Available typelibs:', gi.Repository.get_default().get_loaded_namespaces())"], 
                        capture_output=True, text=True)
                    if result.stdout:
                        print(result.stdout.strip())
                except:
                    pass
        else:
            print(f"\n✓ {module_name} is NOT installed")
    except Exception as e:
        print(f"\n? {module_name} - Error checking: {e}")

# Check what's in the Python path
print("\n\nPython path entries that might contain problematic modules:")
print("=" * 50)
for path in sys.path[:10]:  # First 10 entries
    print(f"  {path}")
EOF

# Method 4: Process runtime analysis (if we can catch it running)
echo ""
echo "Step 5: Attempting runtime process analysis..."
echo "-------------------------------------------"
# Start the app in background
src-tauri/target/debug/airimpute-pro &
APP_PID=$!
sleep 0.5  # Give it a moment to load libraries

# Try to capture loaded libraries
if kill -0 $APP_PID 2>/dev/null; then
    echo "Process $APP_PID is running, capturing loaded libraries..."
    lsof -p $APP_PID 2>/dev/null | grep -E "(\.so|libsoup|webkit)" > "$DIAG_DIR/runtime_libraries.txt" || true
    
    # Also try /proc/maps
    if [ -r "/proc/$APP_PID/maps" ]; then
        grep -E "(libsoup|webkit)" "/proc/$APP_PID/maps" > "$DIAG_DIR/proc_maps.txt" || true
    fi
    
    kill $APP_PID 2>/dev/null || true
else
    echo "Process exited too quickly for runtime analysis"
fi

# Method 5: System configuration check
echo ""
echo "Step 6: Checking system configuration..."
echo "--------------------------------------"
{
    echo "System information:"
    echo "=================="
    uname -a
    echo ""
    echo "Distribution:"
    cat /etc/os-release 2>/dev/null || echo "Unknown"
    echo ""
    echo "Installed libsoup packages:"
    dpkg -l | grep libsoup || echo "No dpkg available or no libsoup packages"
    echo ""
    echo "Available libsoup libraries:"
    find /usr/lib* -name "libsoup*.so*" -type f 2>/dev/null | sort || true
    echo ""
    echo "pkg-config info:"
    pkg-config --list-all | grep soup || echo "No soup in pkg-config"
} > "$DIAG_DIR/system_info.txt"

# Generate summary report
echo ""
echo "Step 7: Generating diagnostic summary..."
echo "-------------------------------------"
{
    echo "AirImpute Pro libsoup Conflict Diagnostic Summary"
    echo "================================================"
    echo "Generated: $(date)"
    echo ""
    
    echo "1. DIRECT EVIDENCE OF CONFLICT:"
    echo "------------------------------"
    if grep -q "libsoup.*3" "$DIAG_DIR/runtime_output.log" 2>/dev/null; then
        echo "✗ libsoup3 conflict detected in runtime output"
        grep "libsoup" "$DIAG_DIR/runtime_output.log" | head -5
    else
        echo "✓ No direct libsoup3 error in runtime output"
    fi
    echo ""
    
    echo "2. FIRST LIBSOUP3 LOAD (from LD_DEBUG):"
    echo "--------------------------------------"
    if [ -s "$DIAG_DIR/libsoup3_first_load.txt" ]; then
        echo "Found libsoup3 load attempts:"
        head -10 "$DIAG_DIR/libsoup3_first_load.txt"
    else
        echo "No libsoup3 loads detected in LD_DEBUG output"
    fi
    echo ""
    
    echo "3. SUSPICIOUS PYTHON MODULES:"
    echo "----------------------------"
    if grep -q "✗" "$DIAG_DIR/python_modules.txt" 2>/dev/null; then
        echo "The following Python modules are installed and may load libsoup3:"
        grep "✗" "$DIAG_DIR/python_modules.txt"
    else
        echo "No suspicious Python modules found"
    fi
    echo ""
    
    echo "4. RECOMMENDATIONS:"
    echo "-----------------"
    echo "Based on the diagnostic results, review the detailed files in $DIAG_DIR/"
    echo "Pay special attention to:"
    echo "  - ld_debug.*.txt: Full dynamic linker trace"
    echo "  - soup_callers.txt: Libraries that request libsoup"
    echo "  - python_modules.txt: Installed Python modules that might cause conflicts"
} > "$DIAG_DIR/SUMMARY.txt"

cat "$DIAG_DIR/SUMMARY.txt"

echo ""
echo "==================================================================="
echo "Diagnostic complete! All results saved to: $DIAG_DIR/"
echo "==================================================================="
echo ""
echo "Next steps:"
echo "1. Review $DIAG_DIR/SUMMARY.txt for the overview"
echo "2. Check $DIAG_DIR/ld_debug.*.txt for detailed loading sequence"
echo "3. Look for 'calling init:' lines before libsoup3 loads"
echo "4. The library mentioned there is your culprit"
echo ""
echo "Share the $DIAG_DIR/ folder with Gemini for targeted solution."