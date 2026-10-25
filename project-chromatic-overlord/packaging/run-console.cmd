@echo off
setlocal
chcp 65001 >nul
if not exist "%~dp0default.zero" (
  echo ERROR: default.zero is missing. Extract the complete ZIP first.
  pause
  exit /b 1
)
echo CHROMATIC experimental UCI engine
echo Type uci, isready, position startpos, go movetime 1000, then quit.
echo For a chessboard interface, register chromatic.exe in a UCI chess GUI.
"%~dp0chromatic.exe" --model "%~dp0default.zero"
set "CHROMATIC_EXIT=%ERRORLEVEL%"
if not "%CHROMATIC_EXIT%"=="0" pause
exit /b %CHROMATIC_EXIT%
