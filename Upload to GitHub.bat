@echo off
title Upload Wink HD Enhancer to GitHub (Private Repo: thuyS)
color 0B
cd /d "C:\Users\Lenovo\.gemini\antigravity\scratch\wink_hd_enhancer"

echo ========================================================
echo   UPLOAD KE GITHUB PRIVATE REPO: thuyS
echo ========================================================
echo.

:: Pastikan path GitHub CLI termuat
set "PATH=%ProgramFiles%\GitHub CLI;%LOCALAPPDATA%\Programs\GitHub CLI;%PATH%"

gh auth status >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [*] Akun GitHub belum terhubung.
    echo [*] Membuka otorisasi login via browser...
    echo.
    gh auth login --web -p https -h github.com
)

echo.
echo [*] Membuat private repository 'thuyS' di GitHub...
gh repo create thuyS --private --source=. --remote=origin --push

echo.
if %ERRORLEVEL% EQU 0 (
    echo ========================================================
    echo  [V] BERHASIL! Repository private 'thuyS' telah dibuat
    echo      dan seluruh file aman berhasil di-push ke GitHub.
    echo ========================================================
) else (
    echo ========================================================
    echo  [!] Jika repository sudah dibuat di web sebelumnya, coba:
    echo      git push -u origin main
    echo ========================================================
)
echo.
pause
