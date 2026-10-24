@echo off
setlocal
set VSLANG=1033
for /f "usebackq tokens=*" %%i in (`"%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do set "MEMORY_VS=%%i"
if not defined MEMORY_VS exit /b 1
call "%MEMORY_VS%\VC\Auxiliary\Build\vcvars64.bat" >nul
if errorlevel 1 exit /b 1
cmake -S "%~dp0." -B "%~dp0build-asan" -G Ninja -DCMAKE_BUILD_TYPE=Release -DCHROMATIC_ASAN=ON "-DCMAKE_CXX_FLAGS_DEBUG=/Od /RTC1 /Z7"
if errorlevel 1 exit /b 1
cmake --build "%~dp0build-asan" --parallel
if errorlevel 1 exit /b 1
ctest --test-dir "%~dp0build-asan" --output-on-failure
exit /b %errorlevel%
