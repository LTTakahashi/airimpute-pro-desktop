# Comprehensive Python DLL Fix for AirImputePro
# PowerShell script with advanced diagnostics and fixes

param(
    [switch]$Diagnose,
    [switch]$Fix,
    [switch]$Rebuild
)

Write-Host "=== Comprehensive AirImputePro Python DLL Fix ===" -ForegroundColor Cyan
Write-Host ""

# Check if running as administrator
$isAdmin = ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole] "Administrator")
if (-not $isAdmin) {
    Write-Host "[WARNING] Not running as administrator. Some operations may fail." -ForegroundColor Yellow
    Write-Host "         Restart PowerShell as Administrator for best results." -ForegroundColor Yellow
    Write-Host ""
}

# Set script location
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $scriptDir

function Test-DllDependencies {
    param([string]$DllPath)
    
    Write-Host "Checking dependencies for: $DllPath" -ForegroundColor Yellow
    
    # Use dumpbin if available
    $dumpbin = Get-Command dumpbin -ErrorAction SilentlyContinue
    if ($dumpbin) {
        & dumpbin /dependents $DllPath
    } else {
        Write-Host "dumpbin not found. Install Visual Studio or Windows SDK for detailed analysis." -ForegroundColor Gray
    }
}

function Find-PythonDlls {
    Write-Host "Searching for Python DLLs..." -ForegroundColor Yellow
    
    $pythonDlls = @()
    $searchPaths = @(
        ".\python-dist",
        ".\src-tauri\python",
        ".\src-tauri\target\release",
        ".\src-tauri\target\debug"
    )
    
    foreach ($path in $searchPaths) {
        if (Test-Path $path) {
            $dlls = Get-ChildItem -Path $path -Filter "python*.dll" -ErrorAction SilentlyContinue
            foreach ($dll in $dlls) {
                $pythonDlls += [PSCustomObject]@{
                    Name = $dll.Name
                    Path = $dll.FullName
                    Size = $dll.Length
                    Version = (Get-Item $dll.FullName).VersionInfo.FileVersion
                }
            }
        }
    }
    
    return $pythonDlls
}

if ($Diagnose -or (-not $Fix -and -not $Rebuild)) {
    Write-Host "=== DIAGNOSTICS ===" -ForegroundColor Green
    Write-Host ""
    
    # Find all Python DLLs
    $pythonDlls = Find-PythonDlls
    Write-Host "Found Python DLLs:" -ForegroundColor Cyan
    $pythonDlls | Format-Table -AutoSize
    
    # Find AirImputePro.exe
    Write-Host "`nSearching for AirImputePro.exe..." -ForegroundColor Yellow
    $exeFiles = Get-ChildItem -Path . -Filter "AirImputePro.exe" -Recurse -ErrorAction SilentlyContinue
    
    if ($exeFiles.Count -gt 0) {
        Write-Host "Found executables:" -ForegroundColor Cyan
        foreach ($exe in $exeFiles) {
            Write-Host "  - $($exe.FullName)" -ForegroundColor White
            
            # Check what DLLs the exe depends on
            if (Get-Command dumpbin -ErrorAction SilentlyContinue) {
                Write-Host "    Dependencies:" -ForegroundColor Gray
                $deps = & dumpbin /dependents $exe.FullName 2>$null | Select-String "python"
                if ($deps) {
                    $deps | ForEach-Object { Write-Host "      $_" -ForegroundColor Gray }
                }
            }
        }
    } else {
        Write-Host "No AirImputePro.exe found!" -ForegroundColor Red
    }
    
    # Check Visual C++ Redistributable
    Write-Host "`nChecking Visual C++ Redistributable..." -ForegroundColor Yellow
    $vcRedist = Get-ItemProperty -Path "HKLM:\SOFTWARE\Microsoft\VisualStudio\14.0\VC\Runtimes\x64" -ErrorAction SilentlyContinue
    if ($vcRedist) {
        Write-Host "[OK] Visual C++ 2015-2022 Redistributable is installed (Version: $($vcRedist.Version))" -ForegroundColor Green
    } else {
        Write-Host "[MISSING] Visual C++ Redistributable not found!" -ForegroundColor Red
        Write-Host "Download from: https://aka.ms/vs/17/release/vc_redist.x64.exe" -ForegroundColor Yellow
    }
    
    # Check Python installation
    Write-Host "`nChecking system Python installations..." -ForegroundColor Yellow
    $pythonVersions = @()
    $pythonPaths = @(
        "$env:LOCALAPPDATA\Programs\Python\Python*",
        "C:\Python*",
        "$env:ProgramFiles\Python*"
    )
    
    foreach ($pattern in $pythonPaths) {
        $dirs = Get-ChildItem -Path $pattern -Directory -ErrorAction SilentlyContinue
        foreach ($dir in $dirs) {
            $pythonExe = Join-Path $dir.FullName "python.exe"
            if (Test-Path $pythonExe) {
                $version = & $pythonExe --version 2>&1
                $pythonVersions += "$version at $($dir.FullName)"
            }
        }
    }
    
    if ($pythonVersions.Count -gt 0) {
        Write-Host "Found Python installations:" -ForegroundColor Cyan
        $pythonVersions | ForEach-Object { Write-Host "  - $_" -ForegroundColor White }
    } else {
        Write-Host "No Python installations found in standard locations" -ForegroundColor Gray
    }
}

if ($Fix) {
    Write-Host "`n=== APPLYING FIXES ===" -ForegroundColor Green
    Write-Host ""
    
    # Fix 1: Create python310.dll from python311.dll
    Write-Host "Fix 1: Creating python310.dll from python311.dll..." -ForegroundColor Yellow
    
    $sourceDll = ".\python-dist\python311.dll"
    $targetDll = ".\python-dist\python310.dll"
    
    if (Test-Path $sourceDll) {
        Copy-Item -Path $sourceDll -Destination $targetDll -Force
        Write-Host "[OK] Created python310.dll in python-dist" -ForegroundColor Green
        
        # Also copy to other locations
        $additionalPaths = @(
            ".\src-tauri\python",
            ".\src-tauri\target\release",
            ".\src-tauri\target\debug"
        )
        
        foreach ($path in $additionalPaths) {
            if (Test-Path $path) {
                Copy-Item -Path $sourceDll -Destination "$path\python310.dll" -Force -ErrorAction SilentlyContinue
                Write-Host "[OK] Copied to $path" -ForegroundColor Green
            }
        }
    } else {
        Write-Host "[SKIP] python311.dll not found" -ForegroundColor Gray
    }
    
    # Fix 2: Copy DLLs next to executable
    Write-Host "`nFix 2: Copying DLLs next to executable..." -ForegroundColor Yellow
    
    $exeFiles = Get-ChildItem -Path . -Filter "AirImputePro.exe" -Recurse -ErrorAction SilentlyContinue
    foreach ($exe in $exeFiles) {
        $exeDir = Split-Path -Parent $exe.FullName
        
        # Copy Python DLLs
        $dllsToCopy = @("python310.dll", "python311.dll", "python3.dll", "vcruntime140.dll", "vcruntime140_1.dll")
        foreach ($dll in $dllsToCopy) {
            $sourcePath = ".\python-dist\$dll"
            if (Test-Path $sourcePath) {
                Copy-Item -Path $sourcePath -Destination $exeDir -Force -ErrorAction SilentlyContinue
                Write-Host "[OK] Copied $dll to $exeDir" -ForegroundColor Green
            }
        }
    }
    
    # Fix 3: Set up symbolic links (requires admin)
    if ($isAdmin) {
        Write-Host "`nFix 3: Creating symbolic links..." -ForegroundColor Yellow
        
        $pythonDistDir = Resolve-Path ".\python-dist" -ErrorAction SilentlyContinue
        if ($pythonDistDir) {
            $exeFiles | ForEach-Object {
                $exeDir = Split-Path -Parent $_.FullName
                $linkPath = Join-Path $exeDir "python-dist"
                
                if (-not (Test-Path $linkPath)) {
                    New-Item -ItemType SymbolicLink -Path $linkPath -Target $pythonDistDir -ErrorAction SilentlyContinue
                    if ($?) {
                        Write-Host "[OK] Created symbolic link in $exeDir" -ForegroundColor Green
                    }
                }
            }
        }
    } else {
        Write-Host "`nFix 3: Skipping symbolic links (requires administrator)" -ForegroundColor Gray
    }
    
    Write-Host "`nFixes applied!" -ForegroundColor Green
}

if ($Rebuild) {
    Write-Host "`n=== REBUILDING APPLICATION ===" -ForegroundColor Green
    Write-Host ""
    
    # Clean build artifacts
    Write-Host "Cleaning build artifacts..." -ForegroundColor Yellow
    if (Test-Path ".\src-tauri\target") {
        Remove-Item -Path ".\src-tauri\target" -Recurse -Force -ErrorAction SilentlyContinue
        Write-Host "[OK] Cleaned target directory" -ForegroundColor Green
    }
    
    # Ensure python310.dll exists before build
    if (-not (Test-Path ".\python-dist\python310.dll")) {
        if (Test-Path ".\python-dist\python311.dll") {
            Copy-Item -Path ".\python-dist\python311.dll" -Destination ".\python-dist\python310.dll" -Force
            Write-Host "[OK] Created python310.dll for build" -ForegroundColor Green
        }
    }
    
    # Run the build
    Write-Host "`nRunning build..." -ForegroundColor Yellow
    Write-Host "This may take several minutes..." -ForegroundColor Gray
    
    & npm run tauri:build:windows
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "`n[SUCCESS] Build completed!" -ForegroundColor Green
        
        # Apply fixes to the built executable
        & $MyInvocation.MyCommand.Path -Fix
    } else {
        Write-Host "`n[ERROR] Build failed!" -ForegroundColor Red
    }
}

# Final instructions
Write-Host "`n=== NEXT STEPS ===" -ForegroundColor Cyan
Write-Host "1. If fixes were applied, try running AirImputePro.exe" -ForegroundColor White
Write-Host "2. If it still fails, run with -Diagnose flag for detailed analysis" -ForegroundColor White
Write-Host "3. Consider running with -Rebuild flag to rebuild the application" -ForegroundColor White
Write-Host "4. Check Windows Event Viewer > Windows Logs > Application for errors" -ForegroundColor White
Write-Host ""
Write-Host "Usage:" -ForegroundColor Yellow
Write-Host "  .\comprehensive-dll-fix.ps1 -Diagnose    # Run diagnostics" -ForegroundColor Gray
Write-Host "  .\comprehensive-dll-fix.ps1 -Fix         # Apply fixes" -ForegroundColor Gray
Write-Host "  .\comprehensive-dll-fix.ps1 -Rebuild     # Rebuild application" -ForegroundColor Gray
Write-Host ""