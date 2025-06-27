@echo off
REM Fix Python DLL issue for AirImputePro.exe on Windows

echo === AirImputePro Python DLL Fix (Windows) ===
echo.

cd /d "%~dp0"

REM Quick fix - Copy python311.dll to python310.dll
echo Applying quick fix...
if exist "src-tauri\python\python311.dll" (
    copy /Y "src-tauri\python\python311.dll" "src-tauri\python\python310.dll" >nul
    echo [OK] Created python310.dll from python311.dll
    
    REM Also copy in Windows bundle directories if they exist
    if exist "src-tauri\target\release\bundle\msi\python311.dll" (
        copy /Y "src-tauri\target\release\bundle\msi\python311.dll" "src-tauri\target\release\bundle\msi\python310.dll" >nul
    )
    if exist "src-tauri\target\release\bundle\nsis\python311.dll" (
        copy /Y "src-tauri\target\release\bundle\nsis\python311.dll" "src-tauri\target\release\bundle\nsis\python310.dll" >nul
    )
) else (
    echo [WARNING] python311.dll not found
)

echo.
echo === Quick fix applied ===
echo.
echo Try running AirImputePro.exe now. If it still doesn't work:
echo.
echo 1. Make sure you're running the .exe from the correct location
echo 2. The python310.dll should be in the same directory as the .exe
echo 3. For a permanent fix, rebuild with matching Python version
echo.
pause