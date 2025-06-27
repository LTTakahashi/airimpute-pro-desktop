@echo off
setlocal enabledelayedexpansion

echo ===================================================
echo AirImpute Pro Desktop - Windows Build Fix
echo ===================================================
echo.

cd /d "%~dp0"

:: Check if we're in the right directory
if not exist "src-tauri" (
    echo ERROR: src-tauri directory not found!
    echo Please run this script from the project root directory.
    exit /b 1
)

cd src-tauri

echo Step 1: Verifying Python installation...
if not exist "python\python.exe" (
    echo ERROR: Python not found in src-tauri\python!
    echo Please ensure Python is properly installed.
    exit /b 1
)

:: Get Python version
for /f "tokens=2" %%i in ('python\python.exe --version 2^>^&1') do set PYTHON_VERSION=%%i
echo Found Python version: %PYTHON_VERSION%

:: Extract major.minor version (e.g., 3.11 from 3.11.9)
for /f "tokens=1,2 delims=." %%a in ("%PYTHON_VERSION%") do (
    set PYTHON_MAJOR=%%a
    set PYTHON_MINOR=%%b
)
set PYTHON_SHORT=%PYTHON_MAJOR%%PYTHON_MINOR%
echo Python short version: %PYTHON_SHORT%

echo.
echo Step 2: Checking for Python DLL...
set PYTHON_DLL=python\python%PYTHON_SHORT%.dll
if exist "%PYTHON_DLL%" (
    echo Found %PYTHON_DLL%
    
    :: Create compatibility copies if needed
    if "%PYTHON_SHORT%"=="311" (
        if not exist "python\python310.dll" (
            echo Creating compatibility copy for python310.dll...
            copy "%PYTHON_DLL%" "python\python310.dll" >nul 2>&1
        )
    )
) else (
    echo WARNING: %PYTHON_DLL% not found!
    echo Checking alternative locations...
    
    :: Check in DLLs directory
    if exist "python\DLLs\python%PYTHON_SHORT%.dll" (
        echo Found in DLLs directory, copying to root...
        copy "python\DLLs\python%PYTHON_SHORT%.dll" "%PYTHON_DLL%" >nul 2>&1
    ) else (
        echo ERROR: Python DLL not found!
        exit /b 1
    )
)

echo.
echo Step 3: Checking for python%PYTHON_SHORT%.lib...
set PYTHON_LIB=python\libs\python%PYTHON_SHORT%.lib
if exist "%PYTHON_LIB%" (
    echo Found %PYTHON_LIB%
) else (
    echo ERROR: %PYTHON_LIB% not found!
    echo PyO3 requires this file for linking.
    exit /b 1
)

echo.
echo Step 4: Checking Visual C++ runtime DLLs...
set VC_DLLS=vcruntime140.dll vcruntime140_1.dll msvcp140.dll
set MISSING_DLLS=0

for %%d in (%VC_DLLS%) do (
    if not exist "python\%%d" (
        echo WARNING: %%d not found in Python directory
        set /a MISSING_DLLS+=1
    )
)

if %MISSING_DLLS% GTR 0 (
    echo Some Visual C++ runtime DLLs are missing.
    echo This might cause runtime issues.
)

echo.
echo Step 5: Using Windows-specific Cargo configuration...
if exist "Cargo-windows.toml" (
    echo Backing up current Cargo.toml...
    copy Cargo.toml Cargo.toml.backup >nul 2>&1
    copy Cargo-windows.toml Cargo.toml >nul 2>&1
    echo Windows Cargo.toml applied.
) else (
    echo WARNING: Cargo-windows.toml not found!
    echo Using default Cargo.toml configuration.
)

echo.
echo Step 6: Setting PyO3 environment variables...
set PYO3_PYTHON=%cd%\python\python.exe
set PYO3_CROSS_LIB_DIR=%cd%\python\libs
set PYO3_CROSS_PYTHON_VERSION=%PYTHON_MAJOR%.%PYTHON_MINOR%
set PYO3_CROSS_PYTHON_IMPLEMENTATION=CPython
set PYTHONHOME=%cd%\python
set PYTHONPATH=%cd%\python;%cd%\python\Lib\site-packages

:: Add Python to PATH
set PATH=%cd%\python;%cd%\python\Scripts;%cd%\python\DLLs;%PATH%

:: Add library path for linking
set LIB=%cd%\python\libs;%cd%\python;%LIB%

echo PyO3 environment configured:
echo   PYO3_PYTHON=%PYO3_PYTHON%
echo   PYO3_CROSS_LIB_DIR=%PYO3_CROSS_LIB_DIR%
echo   PYO3_CROSS_PYTHON_VERSION=%PYO3_CROSS_PYTHON_VERSION%
echo   PYTHONHOME=%PYTHONHOME%

echo.
echo Step 7: Testing Python environment...
"%PYO3_PYTHON%" -c "import sys; print(f'Python {sys.version}')"
if errorlevel 1 (
    echo ERROR: Python test failed!
    exit /b 1
)

echo.
echo Step 8: Cleaning previous builds...
if exist "target" (
    echo Removing old build artifacts...
    cargo clean
)

cd ..

echo.
echo Step 9: Installing npm dependencies...
call npm install --no-audit --no-fund --legacy-peer-deps
if errorlevel 1 (
    echo ERROR: npm install failed!
    exit /b 1
)

:: Ensure Windows-specific modules are installed
echo Checking for Windows-specific modules...
if not exist "node_modules\@rollup\rollup-win32-x64-msvc" (
    echo Installing missing rollup Windows module...
    call npm install @rollup/rollup-win32-x64-msvc --save-optional
)

if not exist "node_modules\@tauri-apps\cli-win32-x64-msvc" (
    echo Installing missing Tauri CLI Windows module...
    call npm install @tauri-apps/cli-win32-x64-msvc --save-optional
)

echo.
echo Step 10: Building frontend...
call npm run build:frontend
if errorlevel 1 (
    echo ERROR: Frontend build failed!
    exit /b 1
)

echo.
echo Step 11: Copying Python DLLs to build directories...
powershell -ExecutionPolicy Bypass -File scripts\copy-python-dlls.ps1
if errorlevel 1 (
    echo WARNING: DLL copy script failed, continuing anyway...
)

echo.
echo Step 12: Building Tauri application...
call npm run tauri build
if errorlevel 1 (
    echo ERROR: Tauri build failed!
    echo.
    echo Common issues:
    echo - Ensure Visual Studio Build Tools are installed
    echo - Check that all Python dependencies are properly installed
    echo - Verify PyO3 can find python%PYTHON_SHORT%.lib
    echo - Try running 'rustup update' to ensure Rust is up to date
    exit /b 1
)

echo.
echo ===================================================
echo Build complete!
echo ===================================================
echo.
echo Installers should be available in:
echo - src-tauri\target\release\bundle\nsis\*.exe
echo - src-tauri\target\release\bundle\msi\*.msi
echo.

:: Restore original Cargo.toml if we modified it
cd src-tauri
if exist "Cargo.toml.backup" (
    echo Restoring original Cargo.toml...
    move /y Cargo.toml.backup Cargo.toml >nul 2>&1
)
cd ..

endlocal