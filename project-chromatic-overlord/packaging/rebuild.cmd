@echo off
setlocal
cmake -S "%~dp0source" -B "%~dp0rebuild" -G "Visual Studio 17 2022" -A x64 -DCHROMATIC_WINDOWS_PORTABLE=ON "-DCHROMATIC_TEST_MODEL=%~dp0default.zero"
if errorlevel 1 exit /b 1
cmake --build "%~dp0rebuild" --config Release --parallel 2
if errorlevel 1 exit /b 1
ctest --test-dir "%~dp0rebuild" -C Release --output-on-failure
exit /b %ERRORLEVEL%
