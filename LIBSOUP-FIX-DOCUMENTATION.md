# LibSoup Conflict - Permanent Fix Documentation

## Problem Summary
The application experiences a critical error when both libsoup2 and libsoup3 are loaded into the same process:
```
(process:2308): libsoup-ERROR **: 16:15:17.020: libsoup3 symbols detected. 
Using libsoup2 and libsoup3 in the same process is not supported.
```

## Root Cause Analysis
1. **Tauri v1.5 Dependency**: Uses `webkit2gtk-4.0` which requires `libsoup2`
2. **Modern Linux Systems**: Ship with `webkit2gtk-4.1` which uses `libsoup3`
3. **Python GObject**: May load libsoup3 through GObject introspection
4. **Symbol Conflict**: GObject type system prevents both versions from coexisting

## Permanent Fix Implementation

### 1. Build-Time Configuration
**File: `.cargo/config.toml`**
- Added Linux-specific target configuration
- Forces linking with libsoup-2.4
- Sets environment variables for build process
- Uses pkg-config wrapper to filter libsoup3

### 2. Build Script Enhancement
**File: `src-tauri/build.rs`**
- Added Linux-specific build configuration
- Sets library search paths
- Forces libsoup2 linking
- Configures WebKit environment variables

### 3. Runtime Scripts
**Created files:**
- `permanent-libsoup-fix.sh` - Main installation script
- `run-dev.sh` - Development launcher with fixes
- `build-production.sh` - Production build script
- `setup-libsoup-env.sh` - Environment configuration
- `pkg-config-wrapper.sh` - Filters libsoup3 references

## Usage Instructions

### Initial Setup
```bash
# Run the permanent fix installer
./permanent-libsoup-fix.sh
```

### Development
```bash
# Use the custom launcher instead of npm run tauri dev
./run-dev.sh

# For clean rebuild
./run-dev.sh --clean
```

### Production Build
```bash
# Use the custom build script
./build-production.sh
```

### Troubleshooting
If issues persist:
1. Run diagnostics: `./diagnose-libsoup-conflict.sh`
2. Check library loading: `LD_DEBUG=libs ./run-dev.sh 2>&1 | grep soup`
3. Verify dependencies: `ldd src-tauri/target/release/airimpute-pro | grep soup`

## Technical Details

### Environment Variables Set
- `LD_PRELOAD`: Forces libsoup2 to load first
- `WEBKIT_DISABLE_COMPOSITING_MODE=1`: Disables hardware compositing
- `WEBKIT_DISABLE_SANDBOX=1`: Disables sandboxing (may reduce conflicts)
- `GTK_USE_PORTAL=1`: Uses portal for file dialogs
- `PKG_CONFIG`: Points to wrapper script

### pkg-config Wrapper
The wrapper filters out libsoup3 references:
- Removes `-lsoup-3.0` flags
- Replaces `libsoup-3.0` with `libsoup-2.4`
- Ensures only libsoup2 is used during compilation

## Alternative Solutions

### 1. Tauri v2 Migration (Recommended Long-term)
```toml
[dependencies]
tauri = "2.0.0-rc"
```
Tauri v2 has better WebKit compatibility.

### 2. Distribution Packaging
- **AppImage**: Bundle webkit2gtk-4.0 and libsoup2
- **Flatpak**: Use org.gnome.Platform//3.38 runtime
- **Snap**: Include all dependencies

### 3. Docker Container
Create a container with compatible libraries:
```dockerfile
FROM ubuntu:20.04
RUN apt-get update && apt-get install -y \
    libwebkit2gtk-4.0-dev \
    libsoup2.4-dev
```

## Verification
The fix is working if:
1. No libsoup error appears on startup
2. `ldd` shows only libsoup-2.4.so.1
3. Application launches successfully
4. WebView renders content properly

## Rollback
To remove the fix:
```bash
./uninstall-libsoup-fix.sh
```
Note: This preserves Cargo.toml and build.rs changes.

## Future Considerations
- Monitor Tauri v2 stability for migration
- Consider static linking for distribution
- Implement CI/CD with proper Linux build environment
- Test on multiple Linux distributions