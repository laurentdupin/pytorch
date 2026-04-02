@echo off
setlocal
set "_CODEX_TEMP=%TEMP%"
set "_CODEX_TMP=%TMP%"
set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
set "VSDEVCMD="
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
if defined _CODEX_TEMP set "TEMP=%_CODEX_TEMP%"
if defined _CODEX_TMP set "TMP=%_CODEX_TMP%"
for %%I in ("%~dp0..\.venv\Scripts\python.exe") do set "PYTHON_EXE=%%~fI"
if not exist "%PYTHON_EXE%" (
  echo Failed to locate repo venv python at "%PYTHON_EXE%">&2
  exit /b 1
)
"%PYTHON_EXE%" %*
exit /b %errorlevel%
