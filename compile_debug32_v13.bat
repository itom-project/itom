TITLE compile debug x86 - %~dp0
CALL "%VS130COMNTOOLS%\vsvars32.bat" x86

msbuild.exe "%~dp0\ALL_BUILD.vcxproj" /p:configuration=debug /p:platform=win32

pause

REM for rebuild add "/t:rebuild" after /p:platform=win32
