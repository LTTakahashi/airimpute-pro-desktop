@echo off
REM Enhanced Python DLL Fix for AirImputePro.exe on Windows
REM This script provides multiple solutions for the python310.dll error

echo === Enhanced AirImputePro Python DLL Fix (Windows) ===
echo.

cd /d "%~dp0"

REM Check if running as administrator
net session >nul 2>&1
if %errorLevel% == 0 (
    echo [OK] Running with administrator privileges
) else (
    echo [WARNING] Not running as administrator. Some fixes may not work.
    echo          Right-click and select "Run as administrator" for best results.
)

echo.
echo Analyzing the issue...
echo.

REM Solution 1: Quick fix - Copy python311.dll to python310.dll
echo Solution 1: Creating python310.dll from python311.dll...
if exist "python-dist\python311.dll" (
    copy /Y "python-dist\python311.dll" "python-dist\python310.dll" >nul 2>&1
    if %errorlevel% == 0 (
        echo [OK] Created python310.dll in python-dist
    ) else (
        echo [FAIL] Could not copy to python-dist (access denied?)
    )
) else (
    echo [SKIP] python311.dll not found in python-dist
)

if exist "src-tauri\python\python311.dll" (
    copy /Y "src-tauri\python\python311.dll" "src-tauri\python\python310.dll" >nul 2>&1
    if %errorlevel% == 0 (
        echo [OK] Created python310.dll in src-tauri\python
    ) else (
        echo [FAIL] Could not copy to src-tauri\python
    )
) else (
    echo [SKIP] python311.dll not found in src-tauri\python
)

REM Also check in release directories
if exist "src-tauri\target\release\python311.dll" (
    copy /Y "src-tauri\target\release\python311.dll" "src-tauri\target\release\python310.dll" >nul 2>&1
    echo [OK] Created python310.dll in release directory
)

REM Solution 2: Copy DLLs next to the executable
echo.
echo Solution 2: Copying Python DLLs next to the executable...

REM Find the AirImputePro.exe
set "EXE_FOUND="
for /f "delims=" %%i in ('where /r . AirImputePro.exe 2^>nul') do (
    set "EXE_PATH=%%i"
    set "EXE_DIR=%%~dpi"
    set "EXE_FOUND=1"
    goto :found_exe
)
:found_exe

if defined EXE_FOUND (
    echo [OK] Found AirImputePro.exe at: %EXE_PATH%
    
    REM Copy python310.dll next to the exe
    if exist "python-dist\python310.dll" (
        copy /Y "python-dist\python310.dll" "%EXE_DIR%" >nul 2>&1
        if %errorlevel% == 0 (
            echo [OK] Copied python310.dll next to executable
        ) else (
            echo [FAIL] Could not copy python310.dll to exe directory
        )
    )
    
    REM Also copy python311.dll just in case
    if exist "python-dist\python311.dll" (
        copy /Y "python-dist\python311.dll" "%EXE_DIR%" >nul 2>&1
        echo [OK] Also copied python311.dll next to executable
    )
    
    REM Copy VC runtime DLLs
    if exist "python-dist\vcruntime140.dll" (
        copy /Y "python-dist\vcruntime140.dll" "%EXE_DIR%" >nul 2>&1
        copy /Y "python-dist\vcruntime140_1.dll" "%EXE_DIR%" >nul 2>&1
        echo [OK] Copied Visual C++ runtime DLLs
    )
) else (
    echo [WARNING] Could not find AirImputePro.exe
)

REM Solution 3: Register Python DLL directory in system PATH (temporary)
echo.
echo Solution 3: Adding Python directory to PATH for this session...
set "PYTHON_DIR=%cd%\python-dist"
if exist "%PYTHON_DIR%" (
    set "PATH=%PYTHON_DIR%;%PATH%"
    echo [OK] Added %PYTHON_DIR% to PATH
) else (
    set "PYTHON_DIR=%cd%\src-tauri\python"
    if exist "%PYTHON_DIR%" (
        set "PATH=%PYTHON_DIR%;%PATH%"
        echo [OK] Added %PYTHON_DIR% to PATH
    )
)

REM Solution 4: Download Python 3.10 if needed
echo.
echo Solution 4: Python 3.10 Download Information
echo If the above solutions don't work, you may need Python 3.10:
echo.
echo Option A: Install Python 3.10 from python.org
echo Option B: Download embeddable package:
echo          https://www.python.org/ftp/python/3.10.11/python-3.10.11-embed-amd64.zip
echo          Extract to: %cd%\python-dist\
echo.

REM Solution 5: Check if Visual C++ Redistributable is installed
echo Solution 5: Checking Visual C++ Redistributable...
reg query "HKLM\SOFTWARE\Microsoft\VisualStudio\14.0\VC\Runtimes\x64" >nul 2>&1
if %errorlevel% == 0 (
    echo [OK] Visual C++ 2015-2022 Redistributable is installed
) else (
    echo [WARNING] Visual C++ Redistributable might not be installed
    echo          Download from: https://aka.ms/vs/17/release/vc_redist.x64.exe
)

echo.
echo === Fixes applied! ===
echo.
echo Try running AirImputePro.exe now. If it still doesn't work:
echo.
echo 1. Run this script as Administrator (right-click, Run as administrator)
echo 2. Install Visual C++ Redistributable if not installed
echo 3. Check Windows Event Viewer for more detailed error messages
echo 4. Rebuild the application with: npm run tauri:build:windows
echo.

REM Try to run the executable if found
if defined EXE_FOUND (
    echo Press any key to try running AirImputePro.exe...
    pause >nul
    start "" "%EXE_PATH%"
) else (
    pause
)