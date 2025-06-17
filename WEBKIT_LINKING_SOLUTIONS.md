# WebKit Linking Solutions for Tauri on Linux

## Problem
The Tauri application fails to build with the error:
```
/usr/bin/ld: cannot find -lwebkit2gtk-4.0: No such file or directory
/usr/bin/ld: cannot find -ljavascriptcoregtk-4.0: No such file or directory
```

This happens because:
- The system has `webkit2gtk-4.1` installed but not `webkit2gtk-4.0`
- Tauri 1.5 (via wry 0.24) expects webkit2gtk-4.0
- The pkg-config for webkit2gtk-4.0 incorrectly points to 4.1 libraries

## Solutions (in order of recommendation)

### 1. Temporary Build Wrapper (Recommended for Quick Fix)
Use the provided build script that creates temporary symlinks:

```bash
./build-with-webkit41.sh        # For release build
./build-with-webkit41.sh dev    # For development mode
```

This script:
- Creates temporary symlinks in `/tmp/webkit-fix-$$`
- Sets `LD_LIBRARY_PATH` to use these symlinks
- Cleans up automatically after build
- Doesn't require system modifications

### 2. System-wide Symbolic Links (Quick but Requires Root)
Create permanent symbolic links:

```bash
sudo ./fix-webkit-symlinks.sh
```

This creates:
- `/usr/lib/x86_64-linux-gnu/libwebkit2gtk-4.0.so` → `libwebkit2gtk-4.1.so.0`
- `/usr/lib/x86_64-linux-gnu/libjavascriptcoregtk-4.0.so` → `libjavascriptcoregtk-4.1.so.0`

To remove later:
```bash
sudo rm /usr/lib/x86_64-linux-gnu/libwebkit2gtk-4.0.so*
sudo rm /usr/lib/x86_64-linux-gnu/libjavascriptcoregtk-4.0.so*
```

### 3. Install webkit2gtk-4.0 (Most Compatible)
Install the actual webkit2gtk-4.0 libraries:

```bash
# For Ubuntu/Debian:
sudo apt install libwebkit2gtk-4.0-37 libjavascriptcoregtk-4.0-18

# For Fedora:
sudo dnf install webkit2gtk3

# For Arch:
sudo pacman -S webkit2gtk
```

### 4. Update to Tauri 2.0 (Best Long-term Solution)
Tauri 2.0 has better support for webkit2gtk-4.1:

1. Update `Cargo.toml`:
```toml
[dependencies]
tauri = "2.0.0-beta"
```

2. Update `package.json`:
```json
"@tauri-apps/api": "^2.0.0-beta",
"@tauri-apps/cli": "^2.0.0-beta"
```

3. Run migration:
```bash
pnpm update @tauri-apps/cli @tauri-apps/api
```

### 5. Use Docker/Podman Container
Build in a container with the correct dependencies:

```dockerfile
FROM rust:1.75
RUN apt-get update && apt-get install -y \
    libwebkit2gtk-4.0-dev \
    libgtk-3-dev \
    libayatana-appindicator3-dev
```

## Verification

After applying any solution, verify it works:

```bash
# Check linking
ldd src-tauri/target/release/airimpute-pro | grep webkit

# Test build
cd src-tauri && cargo build

# Or use pnpm
pnpm tauri build
```

## Troubleshooting

If you still get errors:

1. Clear the build cache:
```bash
cd src-tauri
cargo clean
rm -rf target
```

2. Check pkg-config:
```bash
pkg-config --libs webkit2gtk-4.0
# Should show: -lwebkit2gtk-4.0 (not -lwebkit2gtk-4.1)
```

3. Verify library paths:
```bash
ldconfig -p | grep webkit
```

4. Set explicit library path:
```bash
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
```

## Notes

- The symbolic link solution is safe and reversible
- Different Linux distributions package webkit2gtk differently
- Tauri 2.0 is still in beta but has better Linux support
- Consider using the AppImage format for distribution to avoid dependency issues