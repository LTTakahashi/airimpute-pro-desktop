@echo off
REM Build AirImputePro with Python 3.11 on Windows

echo === Building AirImputePro with Python 3.11 ===
echo.

cd /d "%~dp0"

REM Create python310.dll if needed
if exist "src-tauri\python\python311.dll" (
    if not exist "src-tauri\python\python310.dll" (
        echo Creating python310.dll for compatibility...
        copy /Y "src-tauri\python\python311.dll" "src-tauri\python\python310.dll" >nul
    )
)

REM Set environment variables for PyO3
set "PYO3_PYTHON=%cd%\src-tauri\python\python.exe"
set "PYO3_CROSS_LIB_DIR=%cd%\src-tauri\python"
set "PYO3_CROSS_PYTHON_VERSION=3.11"
set "PYO3_CROSS_PYTHON_IMPLEMENTATION=CPython"
set "PYO3_CROSS=1"

echo Environment variables set:
echo   PYO3_PYTHON=%PYO3_PYTHON%
echo   PYO3_CROSS_LIB_DIR=%PYO3_CROSS_LIB_DIR%
echo   PYO3_CROSS_PYTHON_VERSION=%PYO3_CROSS_PYTHON_VERSION%
echo.

REM Clean previous builds
echo Cleaning previous builds...
cd src-tauri
cargo clean
cd ..

REM Build the application
echo Building application...
call npm run tauri build

echo.
echo Build complete! The executable should now work with the bundled Python.
echo Look for the output in src-tauri\target\release\bundle\
pause