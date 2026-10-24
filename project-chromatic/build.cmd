@echo off
setlocal
chcp 65001 >nul
set VSLANG=1033
if not defined VSCMD_VER (
  for /f "usebackq tokens=*" %%i in (`"%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do set "ZERO_VS=%%i"
)
if not defined VSCMD_VER (
  if not defined ZERO_VS exit /b 1
  call "%ZERO_VS%\VC\Auxiliary\Build\vcvars64.bat" >nul
  if errorlevel 1 exit /b 1
)
cmake -S "%~dp0." -B "%~dp0build" -G Ninja -DCMAKE_BUILD_TYPE=Release "-DCMAKE_CXX_FLAGS_DEBUG=/Od /RTC1 /Z7"
if errorlevel 1 exit /b 1
if "%~1"=="mutation" (
  cmake --build "%~dp0build" --target sign_mutant --parallel
  if errorlevel 1 exit /b 1
  exit /b 0
)
rem A full rebuild also protects against localized MSVC dependency-prefix issues.
cmake --build "%~dp0build" --clean-first --parallel
exit /b %errorlevel%
