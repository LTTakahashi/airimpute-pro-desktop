# Windows Build Guide for AirImpute Pro Desktop

This guide ensures successful builds on Windows, both locally and in GitHub Actions.

## Prerequisites

### Local Development
1. **Windows 10/11** (64-bit)
2. **Visual Studio Build Tools 2022** or Visual Studio 2022
   - Install "Desktop development with C++" workload
   - Ensure Windows 10 SDK is selected
3. **Rust** (latest stable)
   ```powershell
   winget install Rustlang.Rust.MSVC
   ```
4. **Node.js 18+** and npm 9+
5. **Python 3.11.x** (full installation, not embeddable)
   ```powershell
   choco install python311
   ```

### GitHub Actions
All prerequisites are automatically installed by the workflow.

## Quick Start

### Local Build
```powershell
# From project root
.\build-windows-fix.bat
```

### Validate GitHub Actions Setup
```powershell
# Check if everything is configured correctly
.\scripts\validate-github-actions.ps1

# Auto-fix issues (where possible)
.\scripts\validate-github-actions.ps1 -Fix
```

## Architecture Overview

### Python Integration
- Uses **PyO3** for Python-Rust interop
- Embeds Python 3.11 runtime in the application
- Scientific packages: numpy, pandas, scikit-learn, scipy

### Key Components
1. **Frontend**: React + TypeScript + Vite
2. **Backend**: Rust + Tauri
3. **Scientific Computing**: Embedded Python with PyO3
4. **Packaging**: NSIS and MSI installers

## Common Issues and Solutions

### 1. Python DLL Not Found
**Error**: `python311.dll not found`

**Solution**:
- Ensure Python 3.11 is installed (not just embeddable package)
- Run `scripts\copy-python-dlls.ps1` manually
- Check `src-tauri\python\` directory contains the DLL

### 2. PyO3 Linking Errors
**Error**: `error LNK2019: unresolved external symbol`

**Solution**:
- Verify `python311.lib` exists in `src-tauri\python\libs\`
- Check environment variables:
  ```powershell
  echo $env:PYO3_PYTHON
  echo $env:LIB
  ```
- Ensure Visual Studio Build Tools are installed

### 3. Missing Node Modules
**Error**: `Cannot find module '@rollup/rollup-win32-x64-msvc'`

**Solution**:
```powershell
npm install @rollup/rollup-win32-x64-msvc --save-optional
npm install @tauri-apps/cli-win32-x64-msvc --save-optional
```

### 4. GitHub Actions Failures
**Error**: Build succeeds locally but fails in CI

**Check**:
1. Python version mismatch (must be 3.11.x)
2. Missing Visual C++ redistributables
3. Incorrect PyO3 environment variables

## Build Process Details

### 1. Python Setup (GitHub Actions)
```yaml
- Installs Python 3.11.9 via Chocolatey
- Copies entire Python installation to src-tauri/python
- Installs scientific packages
- Configures PyO3 environment
```

### 2. Frontend Build
```bash
npm run build:frontend
```
- Compiles TypeScript
- Bundles with Vite
- Outputs to `dist/` directory

### 3. Tauri Build
```bash
npm run tauri build
```
- Compiles Rust code with embedded Python
- Links against python311.lib
- Packages into NSIS/MSI installers

## Environment Variables

### Required for PyO3
```powershell
$env:PYO3_PYTHON = "path\to\python.exe"
$env:PYO3_CROSS_LIB_DIR = "path\to\python\libs"
$env:PYO3_CROSS_PYTHON_VERSION = "3.11"
$env:PYTHONHOME = "path\to\python"
$env:PYTHONPATH = "path\to\python;path\to\site-packages"
```

### Build Optimization
```powershell
$env:RUST_BACKTRACE = "1"  # For debugging
$env:CARGO_BUILD_JOBS = "4"  # Parallel compilation
```

## Testing the Build

### 1. Verify Python Integration
```powershell
# After build, from src-tauri\target\release
.\airimpute-pro.exe --version
```

### 2. Check Installed Files
The installer should include:
- Main executable
- Python runtime (python311.dll)
- Scientific packages
- Visual C++ runtime DLLs

### 3. Test Installation
1. Run the NSIS installer from `src-tauri\target\release\bundle\nsis\`
2. Launch the application
3. Verify Python features work (try importing data)

## GitHub Actions Secrets

Required secrets in repository settings:
- `TAURI_PRIVATE_KEY`: For update signing (optional)
- `TAURI_KEY_PASSWORD`: Password for private key (optional)

## Troubleshooting Workflow

1. **Run validation script**:
   ```powershell
   .\scripts\validate-github-actions.ps1
   ```

2. **Check build logs**:
   - Look for Python version detection
   - Verify DLL copying steps
   - Check PyO3 environment setup

3. **Enable debug logging**:
   ```yaml
   env:
     RUST_LOG: debug
     RUST_BACKTRACE: 1
   ```

4. **Test minimal build**:
   ```powershell
   cd src-tauri
   cargo build --no-default-features --features custom-protocol
   ```

## Performance Tips

1. **Use Cargo cache in CI**:
   ```yaml
   - uses: Swatinem/rust-cache@v2
   ```

2. **Optimize release builds**:
   - Already configured in Cargo.toml
   - Uses LTO and single codegen unit

3. **Reduce installer size**:
   - Strip debug symbols
   - Compress Python packages
   - Remove unnecessary files

## Security Considerations

1. **Code Signing**: 
   - Use EV certificate for SmartScreen reputation
   - Sign both executable and installer

2. **Python Isolation**:
   - Embedded Python doesn't use system packages
   - Controlled environment variables
   - No user site-packages

3. **DLL Security**:
   - SetDefaultDllDirectories called on startup
   - Only loads DLLs from secure locations

## Maintenance

### Updating Python Version
1. Update version in:
   - `.github/workflows/build-windows.yml`
   - `src-tauri/src/python/runtime_init.rs`
   - Documentation

2. Test with new version:
   ```powershell
   choco install python312  # Example for 3.12
   ```

3. Update package versions in workflow

### Updating Dependencies
```powershell
# Update npm packages
npm update
npm audit fix

# Update Rust dependencies
cd src-tauri
cargo update
```

## Support

For build issues:
1. Check this guide first
2. Run validation script
3. Review GitHub Actions logs
4. Open issue with full error output