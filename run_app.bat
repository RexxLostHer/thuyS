@echo off
title Wink HD Enhancer

:: ============================================================================
::                       WINK HD ENHANCER LAUNCHER
::                 AI Image & Video Super-Resolution Studio
:: ============================================================================

:: 1. Navigate to project root directory
set "PROJECT_DIR=%~dp0"
if "%PROJECT_DIR:~-1%"=="\" set "PROJECT_DIR=%PROJECT_DIR:~0,-1%"
cd /d "%PROJECT_DIR%"
if errorlevel 1 (
    echo [ERROR] Unable to navigate to project root directory: "%PROJECT_DIR%"
    pause
    exit /b 1
)

:: 2. Prepend WinGet FFmpeg binary directory to PATH
set "PATH=C:\Users\Lenovo\AppData\Local\Microsoft\WinGet\Packages\yt-dlp.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-N-127043-g5a54fcf75e-win64-gpl\bin;%PATH%"

:: 3. Verify virtual environment exists (.venv\Scripts\python.exe); bootstrap if missing
if not exist ".venv\Scripts\python.exe" (
    echo [*] Virtual environment missing. Bootstrapping environment...
    if exist "scripts\bootstrap_env.py" (
        where py.exe >nul 2>&1
        if not errorlevel 1 (
            py -3 scripts\bootstrap_env.py
        ) else (
            python scripts\bootstrap_env.py
        )
        if errorlevel 1 (
            echo [ERROR] Failed to bootstrap Python environment.
            pause
            exit /b 1
        )
    ) else (
        echo [ERROR] Bootstrap script scripts\bootstrap_env.py not found.
        pause
        exit /b 1
    )
)

:: 4. Check if server is already running on port 8484
netstat -ano | findstr /R /C:":8484 .*LISTENING" >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo [*] Wink HD Enhancer server is already active on port 8484.
    goto launch_ui
)

:: 5. Launch Uvicorn server in background
echo [*] Starting Wink HD Enhancer backend on http://127.0.0.1:8484 ...
start "" /B ".venv\Scripts\python.exe" -m uvicorn app.main:app --host 127.0.0.1 --port 8484

:: 6. Wait 2 seconds for server initialization
timeout /t 2 /nobreak >nul 2>&1 || powershell -nop -c "Start-Sleep -Seconds 2"

:launch_ui
:: 7. Launch UI in Chrome / Edge App Mode, or fallback to default browser
echo [*] Launching user interface...
if exist "C:\Program Files (x86)\Google\Chrome\Application\chrome.exe" (
    start "" "C:\Program Files (x86)\Google\Chrome\Application\chrome.exe" --disable-http-cache --disable-direct-composition-video-overlays --app=http://127.0.0.1:8484
) else if exist "C:\Program Files\Google\Chrome\Application\chrome.exe" (
    start "" "C:\Program Files\Google\Chrome\Application\chrome.exe" --disable-http-cache --disable-direct-composition-video-overlays --app=http://127.0.0.1:8484
) else if exist "C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe" (
    start "" "C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe" --disable-http-cache --disable-direct-composition-video-overlays --app=http://127.0.0.1:8484
) else if exist "C:\Program Files\Microsoft\Edge\Application\msedge.exe" (
    start "" "C:\Program Files\Microsoft\Edge\Application\msedge.exe" --disable-http-cache --disable-direct-composition-video-overlays --app=http://127.0.0.1:8484
) else (
    start http://127.0.0.1:8484
)

exit /b 0
