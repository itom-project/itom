@echo off
if "%VSWHERE%"=="" set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"

for /f "usebackq tokens=*" %%i in (`"%VSWHERE%" -latest -products * -requires Microsoft.Component.MSBuild -property installationPath`) do (
  set InstallDir=%%i
)

if exist "%InstallDir%\MSBuild\15.0\Bin\MSBuild.exe" (
    @echo on
    TITLE compile release x64 - %~dp0
    CALL "%InstallDir%\Common7\Tools\VsDevCmd.bat"

    msbuild.exe "%~dp0\ALL_BUILD.vcxproj" /p:configuration=release /p:platform=x64
) else (
    echo "Could not find MS VS build tool path"
)
pause

REM for rebuild add "/t:rebuild" after /p:platform=win32

