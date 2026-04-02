@echo off
setlocal
set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
set "VSDEVCMD="
set "CMAKE_EXE="
if exist "%VSWHERE%" (
  for /f "usebackq delims=" %%I in (`"%VSWHERE%" -latest -products * -requires Microsoft.Component.MSBuild -property installationPath`) do (
    set "VSDEVCMD=%%~fI\Common7\Tools\VsDevCmd.bat"
  )
)
if not defined VSDEVCMD (
  if defined VSINSTALLDIR set "VSDEVCMD=%VSINSTALLDIR%\Common7\Tools\VsDevCmd.bat"
)
if not defined VSDEVCMD (
  echo Failed to locate VsDevCmd.bat>&2
  exit /b 1
)
call "%VSDEVCMD%" -arch=x64 -host_arch=x64 >nul
if errorlevel 1 exit /b %errorlevel%
for %%I in (cmake.exe) do set "CMAKE_EXE=%%~$PATH:I"
if not defined CMAKE_EXE (
  if exist "%ProgramFiles%\CMake\bin\cmake.exe" set "CMAKE_EXE=%ProgramFiles%\CMake\bin\cmake.exe"
)
if not defined CMAKE_EXE (
  echo Failed to locate cmake.exe>&2
  exit /b 1
)
"%CMAKE_EXE%" %*
exit /b %errorlevel%
