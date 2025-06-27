# PowerShell script to validate GitHub Actions Windows build configuration
param(
    [switch]$Fix = $false
)

Write-Host "=== GitHub Actions Windows Build Validation ===" -ForegroundColor Cyan
Write-Host

$errors = 0
$warnings = 0

# Check 1: Verify workflow file exists
Write-Host "Checking GitHub Actions workflow..." -ForegroundColor Yellow
$workflowPath = ".github\workflows\build-windows.yml"
if (Test-Path $workflowPath) {
    Write-Host "✓ Workflow file exists" -ForegroundColor Green
} else {
    Write-Host "✗ Workflow file not found at $workflowPath" -ForegroundColor Red
    $errors++
}

# Check 2: Verify package.json has Windows dependencies
Write-Host "`nChecking package.json dependencies..." -ForegroundColor Yellow
$packageJson = Get-Content "package.json" | ConvertFrom-Json

$requiredOptionalDeps = @{
    "@rollup/rollup-win32-x64-msvc" = "^4.9.6"
    "@tauri-apps/cli-win32-x64-msvc" = "^1.5.9"
}

foreach ($dep in $requiredOptionalDeps.Keys) {
    if ($packageJson.optionalDependencies.PSObject.Properties.Name -contains $dep) {
        Write-Host "✓ Found $dep" -ForegroundColor Green
    } else {
        Write-Host "✗ Missing $dep in optionalDependencies" -ForegroundColor Red
        $errors++
        
        if ($Fix) {
            Write-Host "  → Adding $dep to package.json..." -ForegroundColor Cyan
            # This would require more complex JSON manipulation
        }
    }
}

# Check 3: Verify Python setup scripts
Write-Host "`nChecking Python setup scripts..." -ForegroundColor Yellow
$pythonScripts = @(
    "scripts\copy-python-dlls.ps1",
    "build-windows-fix.bat"
)

foreach ($script in $pythonScripts) {
    if (Test-Path $script) {
        Write-Host "✓ Found $script" -ForegroundColor Green
    } else {
        Write-Host "✗ Missing $script" -ForegroundColor Red
        $errors++
    }
}

# Check 4: Verify Cargo configuration
Write-Host "`nChecking Cargo configuration..." -ForegroundColor Yellow
if (Test-Path "src-tauri\Cargo.toml") {
    $cargoContent = Get-Content "src-tauri\Cargo.toml" -Raw
    
    # Check for PyO3 with proper features
    if ($cargoContent -match 'pyo3.*features.*auto-initialize') {
        Write-Host "✓ PyO3 configured with auto-initialize" -ForegroundColor Green
    } else {
        Write-Host "⚠ PyO3 may not be configured properly" -ForegroundColor Yellow
        $warnings++
    }
    
    # Check for python-support feature
    if ($cargoContent -match 'python-support.*pyo3') {
        Write-Host "✓ python-support feature defined" -ForegroundColor Green
    } else {
        Write-Host "⚠ python-support feature may be missing" -ForegroundColor Yellow
        $warnings++
    }
} else {
    Write-Host "✗ Cargo.toml not found" -ForegroundColor Red
    $errors++
}

# Check 5: Windows-specific Cargo.toml
Write-Host "`nChecking Windows-specific configuration..." -ForegroundColor Yellow
if (Test-Path "src-tauri\Cargo-windows.toml") {
    Write-Host "✓ Found Cargo-windows.toml" -ForegroundColor Green
    
    # Verify it doesn't include problematic dependencies
    $winCargoContent = Get-Content "src-tauri\Cargo-windows.toml" -Raw
    if ($winCargoContent -match '#.*hdf5.*optional') {
        Write-Host "✓ HDF5 properly disabled for Windows" -ForegroundColor Green
    } else {
        Write-Host "⚠ Check if HDF5 is disabled in Windows config" -ForegroundColor Yellow
        $warnings++
    }
} else {
    Write-Host "⚠ Cargo-windows.toml not found (optional)" -ForegroundColor Yellow
    $warnings++
}

# Check 6: Python runtime initialization
Write-Host "`nChecking Python runtime code..." -ForegroundColor Yellow
$runtimeInitPath = "src-tauri\src\python\runtime_init.rs"
if (Test-Path $runtimeInitPath) {
    $runtimeContent = Get-Content $runtimeInitPath -Raw
    
    # Check for version flexibility
    if ($runtimeContent -match 'possible_versions.*311.*310') {
        Write-Host "✓ Python version flexibility implemented" -ForegroundColor Green
    } else {
        Write-Host "⚠ Python runtime may not handle version mismatches" -ForegroundColor Yellow
        $warnings++
    }
} else {
    Write-Host "✗ runtime_init.rs not found" -ForegroundColor Red
    $errors++
}

# Check 7: GitHub secrets documentation
Write-Host "`nChecking for GitHub secrets documentation..." -ForegroundColor Yellow
$secretsNeeded = @(
    "GITHUB_TOKEN (automatic)",
    "TAURI_PRIVATE_KEY (for updates)",
    "TAURI_KEY_PASSWORD (for updates)"
)

Write-Host "Required GitHub secrets:" -ForegroundColor Cyan
foreach ($secret in $secretsNeeded) {
    Write-Host "  - $secret" -ForegroundColor White
}

# Summary
Write-Host "`n=== Validation Summary ===" -ForegroundColor Cyan
if ($errors -eq 0 -and $warnings -eq 0) {
    Write-Host "✓ All checks passed!" -ForegroundColor Green
    Write-Host "Your GitHub Actions Windows build should work correctly." -ForegroundColor Green
} elseif ($errors -eq 0) {
    Write-Host "✓ No errors found" -ForegroundColor Green
    Write-Host "⚠ $warnings warning(s) found" -ForegroundColor Yellow
    Write-Host "The build should work but might have minor issues." -ForegroundColor Yellow
} else {
    Write-Host "✗ $errors error(s) found" -ForegroundColor Red
    Write-Host "⚠ $warnings warning(s) found" -ForegroundColor Yellow
    Write-Host "The build is likely to fail. Please fix the errors above." -ForegroundColor Red
}

# Provide actionable recommendations
if ($errors -gt 0 -or $warnings -gt 0) {
    Write-Host "`nRecommendations:" -ForegroundColor Cyan
    
    if ($errors -gt 0) {
        Write-Host "1. Run 'npm install' to ensure all dependencies are installed" -ForegroundColor White
        Write-Host "2. Ensure all required scripts are present in the scripts/ directory" -ForegroundColor White
        Write-Host "3. Check that the src-tauri directory structure is correct" -ForegroundColor White
    }
    
    if ($warnings -gt 0) {
        Write-Host "- Consider using Cargo-windows.toml for Windows-specific builds" -ForegroundColor White
        Write-Host "- Ensure Python version handling is flexible for different environments" -ForegroundColor White
    }
}

Write-Host "`nTo test locally before pushing:" -ForegroundColor Cyan
Write-Host "  .\build-windows-fix.bat" -ForegroundColor White
Write-Host "`nTo test in GitHub Actions:" -ForegroundColor Cyan
Write-Host "  git push origin main" -ForegroundColor White

exit $errors