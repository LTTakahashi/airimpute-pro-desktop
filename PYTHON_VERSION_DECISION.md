# Python Version Decision for Windows Build

## Issue Summary
The GitHub Actions workflow was failing because Python 3.10.13 embeddable package doesn't exist on Python.org's FTP server.

## Root Cause Analysis
After extensive investigation with Gemini's help:
1. Python 3.10.13 was a security release
2. Non-essential build artifacts like the embeddable zip are sometimes omitted during rapid patch cycles
3. The file `python-3.10.13-embed-amd64.zip` is confirmed to not exist on the FTP server

## Solution Implemented
We switched to Python 3.10.11, which:
- Has a confirmed working embeddable package on Python.org
- Is compatible with all our dependencies (numpy, pandas, scikit-learn, etc.)
- Works with PyO3 0.20
- Maintains the same minor version (3.10.x) ensuring compatibility

## Alternative Solutions Considered
1. **Self-hosting the artifact** - Most robust but adds complexity
2. **Using NuGet** - Adds another dependency
3. **Using different Python version** - Implemented (3.10.11)

## Future Recommendations
If a specific Python version is absolutely required in the future:
1. Download the embeddable package from NuGet (python package)
2. Verify its SHA256 hash
3. Host it in GitHub Releases or another stable location
4. Update the workflow to download from the stable URL with checksum verification

## Verification
The embeddable package URL for 3.10.11 can be verified to exist:
```
https://www.python.org/ftp/python/3.10.11/python-3.10.11-embed-amd64.zip
```

This ensures our CI/CD pipeline remains stable and doesn't depend on potentially missing artifacts.