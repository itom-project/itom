CALL "%VS100COMNTOOLS%\vsvars32.bat" x86

msbuild.exe ALL_BUILD.vcxproj /p:configuration=release /p:platform=win32

pause

REM for rebuild add "/t:rebuild" after /p:platform=win32
