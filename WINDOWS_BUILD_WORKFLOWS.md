# Windows Build Workflows

This project provides three different GitHub Actions workflows for building Windows executables:

## 1. Native Windows Build (Recommended)

**File:** `.github/workflows/build-windows-native.yml`

- **Runner:** `windows-latest`
- **Target:** `x86_64-pc-windows-msvc`
- **Advantages:**
  - Most reliable and compatible
  - Native MSVC toolchain
  - Official Tauri build path
  - Produces smallest executables
  - Best performance
- **Disadvantages:**
  - Requires Windows runner (slightly more expensive)

## 2. MinGW Cross-Compilation

**File:** `.github/workflows/build-windows-mingw.yml`

- **Runner:** `ubuntu-22.04`
- **Target:** `x86_64-pc-windows-gnu`
- **Advantages:**
  - Builds on Linux (cheaper runners)
  - Simpler cross-compilation setup
  - No Windows license required
- **Disadvantages:**
  - Larger executables (includes MinGW runtime)
  - Potential compatibility issues with some Windows APIs
  - Different C runtime (not MSVC)

## 3. Zig Cross-Compilation (Currently Broken)

**File:** `.github/workflows/build-windows.yml`

- **Status:** Not working - cargo-zigbuild doesn't support Windows targets
- **Issue:** Attempts to use MSVC linker on Linux which doesn't exist

## How to Use

### Running a Workflow

1. Go to the Actions tab in your GitHub repository
2. Select the workflow you want to run:
   - "Build Windows Executable (Native)" - Recommended
   - "Build Windows Executable (MinGW Cross-Compile)" - Alternative
3. Click "Run workflow"
4. Choose options:
   - **Build type:** `release` (optimized) or `debug` (with debug symbols)
   - **Version:** Version number for the build (e.g., `1.0.0`)
5. Click "Run workflow" button

### Downloading Artifacts

After the workflow completes:

1. Go to the workflow run page
2. Scroll to "Artifacts" section
3. Download the artifact (e.g., `windows-build-release-1.0.0`)
4. Extract the ZIP file containing:
   - Installer executable (.exe)
   - MSI installer (native build only)
   - SHA256 checksums
   - Build information

## Choosing the Right Workflow

### Use Native Windows Build when:
- You need maximum compatibility
- You want the smallest executable size
- You're building for production release
- You need MSVC-specific features

### Use MinGW Cross-Compilation when:
- You must build on Linux infrastructure
- You're okay with larger executables
- You don't need MSVC-specific features
- Cost is a primary concern

## Troubleshooting

### Native Build Issues
- Ensure all Windows SDK components are available
- Check that WebView2 runtime is properly configured
- Verify Python embeddable package download

### MinGW Build Issues
- Some Windows APIs might not work correctly
- Antivirus software may flag MinGW executables
- Runtime DLLs must be bundled with the application

### General Issues
- Check workflow logs for specific errors
- Ensure all dependencies are properly versioned
- Verify GitHub secrets are set if using code signing