@echo off
REM Comprehensive Windows build fix for AirImputePro
REM This script ensures Python DLLs are properly configured

echo === AirImputePro Windows Build Fix ===
echo.

cd /d "%~dp0"

REM Step 1: Ensure python310.dll exists (copy from python311.dll)
echo [1/5] Checking Python DLL compatibility...
if exist "src-tauri\python\python311.dll" (
    if not exist "src-tauri\python\python310.dll" (
        echo Creating python310.dll from python311.dll...
        copy /Y "src-tauri\python\python311.dll" "src-tauri\python\python310.dll" >nul
        echo [OK] Created python310.dll
    ) else (
        echo [OK] python310.dll already exists
    )
) else (
    echo [ERROR] python311.dll not found!
    pause
    exit /b 1
)

REM Step 2: Set PyO3 environment variables
echo.
echo [2/5] Setting PyO3 environment variables...
set "PYO3_PYTHON=%cd%\src-tauri\python\python.exe"
set "PYO3_CROSS_LIB_DIR=%cd%\src-tauri\python"
set "PYO3_CROSS_PYTHON_VERSION=3.11"
set "PYO3_CROSS_PYTHON_IMPLEMENTATION=CPython"
set "PYO3_CROSS=1"

echo PYO3_PYTHON=%PYO3_PYTHON%
echo PYO3_CROSS_LIB_DIR=%PYO3_CROSS_LIB_DIR%
echo PYO3_CROSS_PYTHON_VERSION=%PYO3_CROSS_PYTHON_VERSION%

REM Step 3: Clean previous builds
echo.
echo [3/5] Cleaning previous builds...
cd src-tauri
cargo clean 2>nul
cd ..

REM Step 4: Build the application
echo.
echo [4/5] Building application...
call npm run tauri build

REM Step 5: Post-build DLL copy
echo.
echo [5/5] Post-build DLL verification...
set "BUILD_DIR=src-tauri\target\release"
set "BUNDLE_DIR=src-tauri\target\release\bundle"

REM Copy DLLs to release directory
if exist "%BUILD_DIR%\air-impute-pro.exe" (
    if not exist "%BUILD_DIR%\python310.dll" (
        copy /Y "src-tauri\python\python310.dll" "%BUILD_DIR%\" >nul
        echo [OK] Copied python310.dll to release directory
    )
    if not exist "%BUILD_DIR%\python311.dll" (
        copy /Y "src-tauri\python\python311.dll" "%BUILD_DIR%\" >nul
        echo [OK] Copied python311.dll to release directory
    )
)

REM Copy DLLs to Windows installers
for %%D in (msi nsis) do (
    if exist "%BUNDLE_DIR%\%%D\" (
        copy /Y "src-tauri\python\python310.dll" "%BUNDLE_DIR%\%%D\" >nul 2>&1
        copy /Y "src-tauri\python\python311.dll" "%BUNDLE_DIR%\%%D\" >nul 2>&1
        echo [OK] Copied Python DLLs to %%D bundle
    )
)

echo.
echo === Build Complete ===
echo.
echo The executable should now work with the bundled Python.
echo Look for outputs in:
echo   - Executable: src-tauri\target\release\air-impute-pro.exe
echo   - Installers: src-tauri\target\release\bundle\
echo.
pause