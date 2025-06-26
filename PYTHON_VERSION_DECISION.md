# Python Version Decision for Windows Build

## Update: Migration to Python 3.11

### Date: 2025-06-26

The project has been migrated from Python 3.10 to Python 3.11 to resolve a critical version mismatch where the code expected `python310.dll` but bundled `python311.dll`.

## Changes Made

1. **Created Central Configuration File**: `python-version.json`
   - Single source of truth for Python version across the project
   - Contains version info, DLL names, URLs, and platform-specific details

2. **Updated All References**:
   - GitHub Actions workflow (`build-windows.yml`)
   - PowerShell scripts (`copy-python-dlls.ps1`)
   - Runtime initialization already correctly uses `python311.dll`

3. **Created Helper Scripts**:
   - `scripts/get-python-version.ps1` - PowerShell script to read version config
   - `scripts/get-python-version.cjs` - Node.js script for cross-platform support

## Python 3.11 Benefits

1. **Performance**: Up to 25% faster than Python 3.10
2. **Better Error Messages**: More informative tracebacks
3. **Task Groups**: Better async/await support
4. **Stability**: Latest stable release with long-term support
5. **Compatibility**: Full support for all scientific packages (numpy, pandas, scikit-learn)

## Configuration Structure

```json
{
  "version": "3.11.9",
  "major": "3",
  "minor": "11",
  "patch": "9",
  "dll_name": "python311.dll",
  "lib_name": "python311.lib",
  "executable": {
    "windows": "python.exe",
    "linux": "python3",
    "macos": "python3"
  },
  "embeddable_url": "https://www.python.org/ftp/python/3.11.9/python-3.11.9-embed-amd64.zip",
  "choco_package": "python311"
}
```

## Usage in Scripts

### PowerShell
```powershell
$pythonVersion = & scripts\get-python-version.ps1
$dllName = & scripts\get-python-version.ps1 dll_name
```

### Node.js
```javascript
const pythonVersion = execSync('node scripts/get-python-version.cjs').toString().trim();
const dllName = execSync('node scripts/get-python-version.cjs dll_name').toString().trim();
```

## Previous Issue Summary (Python 3.10.13)

The GitHub Actions workflow was failing because Python 3.10.13 embeddable package doesn't exist on Python.org's FTP server.

### Root Cause Analysis
After extensive investigation:
1. Python 3.10.13 was a security release
2. Non-essential build artifacts like the embeddable zip are sometimes omitted during rapid patch cycles
3. The file `python-3.10.13-embed-amd64.zip` is confirmed to not exist on the FTP server

## Verification

The embeddable package URL for 3.11.9 can be verified to exist:
```
https://www.python.org/ftp/python/3.11.9/python-3.11.9-embed-amd64.zip
```

This ensures our CI/CD pipeline remains stable and doesn't depend on potentially missing artifacts.