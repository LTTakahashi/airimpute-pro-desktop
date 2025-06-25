# PowerShell script to ensure Python DLLs are copied for Windows builds
param(
    [string]$PythonDir = "src-tauri\python",
    [string]$TargetDir = "src-tauri\target\release"
)

Write-Host "=== Copying Python DLLs for Windows Build ==="

# Check if Python directory exists
if (-not (Test-Path $PythonDir)) {
    Write-Error "Python directory not found: $PythonDir"
    exit 1
}

# Find python310.dll
$pythonDll = Get-ChildItem -Path $PythonDir -Filter "python310.dll" -Recurse -ErrorAction SilentlyContinue | Select-Object -First 1

if ($pythonDll) {
    Write-Host "Found python310.dll at: $($pythonDll.FullName)"
    
    # Copy to target directory if it exists
    if (Test-Path $TargetDir) {
        Copy-Item -Path $pythonDll.FullName -Destination $TargetDir -Force
        Write-Host "✓ Copied python310.dll to $TargetDir"
    }
    
    # Also copy to the root of Python directory if not already there
    $rootDll = Join-Path $PythonDir "python310.dll"
    if (-not (Test-Path $rootDll)) {
        Copy-Item -Path $pythonDll.FullName -Destination $rootDll -Force
        Write-Host "✓ Copied python310.dll to root of Python directory"
    }
} else {
    Write-Error "python310.dll not found in $PythonDir"
    exit 1
}

# Copy other essential DLLs
$essentialDlls = @(
    "vcruntime140.dll",
    "vcruntime140_1.dll",
    "msvcp140.dll"
)

foreach ($dll in $essentialDlls) {
    $found = Get-ChildItem -Path $PythonDir -Filter $dll -Recurse -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($found) {
        Write-Host "Found $dll at: $($found.FullName)"
        if (Test-Path $TargetDir) {
            Copy-Item -Path $found.FullName -Destination $TargetDir -Force
            Write-Host "✓ Copied $dll to $TargetDir"
        }
    } else {
        Write-Warning "$dll not found (may not be required)"
    }
}

Write-Host "`n✓ Python DLL copy complete"