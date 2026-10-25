@echo off
setlocal
chcp 65001 >nul
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0verify-package.ps1"
set "CHROMATIC_EXIT=%ERRORLEVEL%"
if not "%CHROMATIC_EXIT%"=="0" pause
exit /b %CHROMATIC_EXIT%
