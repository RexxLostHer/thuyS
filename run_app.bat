@echo off
setlocal EnableDelayedExpansion
title Wink HD Enhancer

:: ============================================================================
::                       WINK HD ENHANCER LAUNCHER
::                 AI Image & Video Super-Resolution Studio
:: ============================================================================

:: 1. Navigate to project root directory reliably
pushd "%~dp0"
if not exist "app\main.py" (
    cd /d "C:\Users\Lenovo\.gemini\antigravity\scratch\wink_hd_enhancer"
)

:: 2. Prepend WinGet FFmpeg binary directory to PATH
if exist "C:\Users\Lenovo\AppData\Local\Microsoft\WinGet\Packages\yt-dlp.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-N-127043-g5a54fcf75e-win64-gpl\bin" (
    set "PATH=C:\Users\Lenovo\AppData\Local\Microsoft\WinGet\Packages\yt-dlp.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-N-127043-g5a54fcf75e-win64-gpl\bin;%PATH%"
)

:: 3. Select python executable
set "PYTHON_EXE=C:\Users\Lenovo\.gemini\antigravity\scratch\wink_hd_enhancer\.venv\Scripts\python.exe"
if not exist "%PYTHON_EXE%" (
    if exist ".venv\Scripts\python.exe" (
        set "PYTHON_EXE=.venv\Scripts\python.exe"
    ) else (
        where py.exe >nul 2>&1
        if not errorlevel 1 (
            set "PYTHON_EXE=py -3"
        ) else (
            set "PYTHON_EXE=python"
        )
    )
)

:: 4. Check if server is already running on port 8484
netstat -ano | findstr /R /C:":8484 .*LISTENING" >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo [*] Wink HD Enhancer backend is already active on port 8484.
    goto launch_ui
)

:: 5. Launch Uvicorn server in background
echo [*] Starting Wink HD Enhancer backend on http://127.0.0.1:8484 ...
start "" /B "%PYTHON_EXE%" -m uvicorn app.main:app --host 0.0.0.0 --port 8484

:: 6. Wait 2 seconds for server initialization
timeout /t 2 /nobreak >nul 2>&1 || powershell -nop -c "Start-Sleep -Seconds 2"

:launch_ui
:: 7. Launch UI in Chrome / Edge App Mode, or fallback to default browser
echo [*] Launching user interface...
if exist "C:\Program Files\Google\Chrome\Application\chrome.exe" (
    start "" "C:\Program Files\Google\Chrome\Application\chrome.exe" --disable-http-cache --disable-direct-composition-video-overlays --app=http://127.0.0.1:8484
) else if exist "C:\Program Files (x86)\Google\Chrome\Application\chrome.exe" (
    start "" "C:\Program Files (x86)\Google\Chrome\Application\chrome.exe" --disable-http-cache --disable-direct-composition-video-overlays --app=http://127.0.0.1:8484
) else if exist "C:\Program Files\Microsoft\Edge\Application\msedge.exe" (
    start "" "C:\Program Files\Microsoft\Edge\Application\msedge.exe" --disable-http-cache --disable-direct-composition-video-overlays --app=http://127.0.0.1:8484
) else if exist "C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe" (
    start "" "C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe" --disable-http-cache --disable-direct-composition-video-overlays --app=http://127.0.0.1:8484
) else (
    start http://127.0.0.1:8484
)

popd
exit /b 0
