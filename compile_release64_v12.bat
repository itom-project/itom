CALL "%VS120COMNTOOLS%\vsvars32.bat" x64

msbuild.exe %~dp0\ALL_BUILD.vcxproj /p:configuration=release /p:platform=x64

pause

REM for rebuild add "/t:rebuild" after /p:platform=win32