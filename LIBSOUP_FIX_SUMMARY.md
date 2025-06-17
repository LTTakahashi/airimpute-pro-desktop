# AirImpute Pro - libsoup2/libsoup3 Conflict Resolution

## Problem Summary

The application fails to start with the error:
```
libsoup-ERROR: libsoup3 symbols detected. Using libsoup2 and libsoup3 in the same process is not supported.
```

## Root Cause Analysis (via Gemini)

1. **webkit2gtk-4.1** on the system is compiled against libsoup3
2. The application expects webkit2gtk-4.0 which uses libsoup2
3. When both libraries load in the same process, they conflict due to:
   - GObject type name collisions
   - Incompatible ABI
   - Shared global state in GLib

## Implemented Solution: Combined Arms Approach

### 1. Build-time Control (pkg-config-wrapper.sh)
- Filters out `-lsoup-3.0` from pkg-config output
- Prevents our application from directly linking against libsoup3
- Ensures our code uses libsoup2 symbols

### 2. Runtime Control (launch-airimpute.sh)
- Uses `LD_PRELOAD` to force libsoup2 to load first
- Creates webkit 4.0→4.1 compatibility layer
- Provides fixed .pc files that reference libsoup-2.4

### 3. Python GI Version Fix (gi_version_fix.py)
- Forces GObject Introspection to use Soup 2.4
- Prevents Python code from accidentally loading libsoup3
- Imported early in Python initialization

## How to Run the Application

```bash
./launch-airimpute.sh
```

This script:
1. Sets up the pkg-config wrapper
2. Configures LD_PRELOAD for libsoup2
3. Creates webkit compatibility layer
4. Runs `npm run tauri dev` with the fixed environment

## What to Watch For

⚠️ **Potential Issues:**
- GType warnings about duplicate type registration
- Crashes in webkit or soup functions
- TLS handshake failures
- Strange GLib main loop behavior

If you see: `GType-WARNING **: ... already registered`
This means the conflict is too severe and requires recompilation.

## Alternative Solutions (if this fails)

1. **Container/Flatpak**: Bundle the app with its own webkit2gtk-4.0
2. **Static Linking**: Rebuild with statically linked libraries
3. **Downgrade System**: Use Ubuntu 22.04 which has webkit2gtk-4.0
4. **Migrate to libsoup3**: Update the entire application stack

## Technical Details

### Files Created:
- `pkg-config-wrapper.sh` - Filters pkg-config output
- `launch-airimpute.sh` - Combined launcher script
- `gi_version_fix.py` - Python GI version forcing
- Various other test scripts

### Environment Variables Used:
- `LD_PRELOAD` - Force libsoup2 loading
- `PKG_CONFIG` - Use our wrapper
- `PKG_CONFIG_PATH` - Add webkit fix directory
- `WEBKIT_DISABLE_COMPOSITING_MODE=1` - Avoid webkit issues

## Testing

Run the test script to verify Python fix:
```bash
python3 test-libsoup-fix.py
```

## Status

The solution has been implemented and the application is building with the fixes applied. The pkg-config wrapper successfully filters out libsoup3 references, showing `-lsoup-2.4` in the output instead.