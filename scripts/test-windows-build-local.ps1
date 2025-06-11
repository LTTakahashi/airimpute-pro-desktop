# PowerShell script to test Windows build locally
# This replicates the GitHub Actions environment as closely as possible

Write-Host "=== Testing Windows Build Locally ===" -ForegroundColor Cyan
Write-Host ""

# Check if running on Windows
if ($env:OS -ne "Windows_NT") {
    Write-Error "This script must be run on Windows"
    exit 1
}

# Function to check command availability
function Test-Command {
    param($Command)
    try {
        Get-Command $Command -ErrorAction Stop | Out-Null
        return $true
    } catch {
        return $false
    }
}

# Check prerequisites
Write-Host "Checking prerequisites..." -ForegroundColor Yellow

$missing = @()

if (-not (Test-Command "node")) {
    $missing += "Node.js"
}

if (-not (Test-Command "npm")) {
    $missing += "npm"
}

if (-not (Test-Command "rustup")) {
    $missing += "Rust"
}

if (-not (Test-Command "cargo")) {
    $missing += "Cargo"
}

$pythonCmd = $null
foreach ($cmd in @("python", "python3", "py")) {
    if (Test-Command $cmd) {
        $pythonCmd = $cmd
        break
    }
}
if (-not $pythonCmd) {
    $missing += "Python"
}

if ($missing.Count -gt 0) {
    Write-Error "Missing prerequisites: $($missing -join ', ')"
    Write-Host ""
    Write-Host "Please install missing components:" -ForegroundColor Yellow
    Write-Host "- Node.js: https://nodejs.org/"
    Write-Host "- Rust: https://rustup.rs/"
    Write-Host "- Python: https://www.python.org/"
    exit 1
}

Write-Host "✓ All prerequisites found" -ForegroundColor Green
Write-Host ""

# Display versions
Write-Host "Environment:" -ForegroundColor Yellow
node --version
npm --version
rustup --version
cargo --version
& $pythonCmd --version
Write-Host ""

# Configure npm for Windows
Write-Host "Configuring npm for Windows build..." -ForegroundColor Yellow
npm config set msvs_version 2022
npm config set python $pythonCmd
Write-Host "✓ npm configured" -ForegroundColor Green
Write-Host ""

# Clean previous build
Write-Host "Cleaning previous build artifacts..." -ForegroundColor Yellow
if (Test-Path "node_modules") {
    Remove-Item -Path "node_modules" -Recurse -Force
}
if (Test-Path "package-lock.json") {
    Remove-Item -Path "package-lock.json" -Force
}
if (Test-Path "dist") {
    Remove-Item -Path "dist" -Recurse -Force
}
if (Test-Path "src-tauri/target") {
    Write-Host "  Cleaning Rust target (this may take a while)..."
    Remove-Item -Path "src-tauri/target" -Recurse -Force
}
Write-Host "✓ Clean complete" -ForegroundColor Green
Write-Host ""

# Install dependencies
Write-Host "Installing dependencies..." -ForegroundColor Yellow
$installStart = Get-Date
npm install --no-audit --no-fund --loglevel verbose 2>&1 | Tee-Object -FilePath npm-install-local.log

if ($LASTEXITCODE -ne 0) {
    Write-Error "npm install failed with exit code $LASTEXITCODE"
    Write-Host ""
    Write-Host "Check npm-install-local.log for details" -ForegroundColor Yellow
    Get-Content npm-install-local.log | Select-Object -Last 30
    exit 1
}

$installTime = (Get-Date) - $installStart
Write-Host "✓ Dependencies installed in $($installTime.TotalSeconds) seconds" -ForegroundColor Green
Write-Host ""

# Build frontend
Write-Host "Building frontend..." -ForegroundColor Yellow
$env:NODE_OPTIONS = "--max-old-space-size=8192"
$buildStart = Get-Date

npm run build

if ($LASTEXITCODE -ne 0) {
    Write-Error "Frontend build failed"
    exit 1
}

$buildTime = (Get-Date) - $buildStart
Write-Host "✓ Frontend built in $($buildTime.TotalSeconds) seconds" -ForegroundColor Green
Write-Host ""

# Check Rust target
Write-Host "Checking Rust Windows target..." -ForegroundColor Yellow
$targets = rustup target list --installed
if (-not ($targets -match "x86_64-pc-windows-msvc")) {
    Write-Host "Installing x86_64-pc-windows-msvc target..."
    rustup target add x86_64-pc-windows-msvc
}
Write-Host "✓ Rust target ready" -ForegroundColor Green
Write-Host ""

# Build Tauri application
Write-Host "Building Tauri application..." -ForegroundColor Yellow
Write-Host "This may take several minutes on first build..." -ForegroundColor Cyan
$tauriStart = Get-Date

npm run tauri build -- --target x86_64-pc-windows-msvc

if ($LASTEXITCODE -ne 0) {
    Write-Error "Tauri build failed"
    exit 1
}

$tauriTime = (Get-Date) - $tauriStart
Write-Host "✓ Tauri built in $($tauriTime.TotalMinutes) minutes" -ForegroundColor Green
Write-Host ""

# List artifacts
Write-Host "Build artifacts:" -ForegroundColor Yellow
Write-Host ""

$msiPath = "src-tauri\target\x86_64-pc-windows-msvc\release\bundle\msi"
$nsisPath = "src-tauri\target\x86_64-pc-windows-msvc\release\bundle\nsis"
$exePath = "src-tauri\target\x86_64-pc-windows-msvc\release"

if (Test-Path $msiPath) {
    Get-ChildItem -Path $msiPath -Filter "*.msi" | ForEach-Object {
        Write-Host "  MSI: $($_.Name) ($([math]::Round($_.Length / 1MB, 2)) MB)" -ForegroundColor Green
    }
}

if (Test-Path $nsisPath) {
    Get-ChildItem -Path $nsisPath -Filter "*.exe" | ForEach-Object {
        Write-Host "  NSIS: $($_.Name) ($([math]::Round($_.Length / 1MB, 2)) MB)" -ForegroundColor Green
    }
}

if (Test-Path "$exePath\airimpute-pro-desktop.exe") {
    $exe = Get-Item "$exePath\airimpute-pro-desktop.exe"
    Write-Host "  EXE: $($exe.Name) ($([math]::Round($exe.Length / 1MB, 2)) MB)" -ForegroundColor Green
}

Write-Host ""
Write-Host "=== Build completed successfully! ===" -ForegroundColor Green
Write-Host ""
Write-Host "Total build time: $((Get-Date) - $installStart)" -ForegroundColor Cyan