@echo off

SET /P ANSWER=Debug[1], Release[2], both[3]?
echo You chose: %ANSWER%
if /i {%ANSWER%}=={1} (goto :debug)
if /i {%ANSWER%}=={2} (goto :release)
if /i {%ANSWER%}=={3} (goto :debug)

echo Wrong input.
goto :end

:debug
CALL "%VS120COMNTOOLS%\vsvars32.bat" x86

echo Compile Debug...
msbuild.exe itom\ALL_BUILD.vcxproj /p:configuration=debug /p:platform=x64
msbuild.exe designerplugins\ALL_BUILD.vcxproj /p:configuration=debug /p:platform=x64
msbuild.exe plugins\ALL_BUILD.vcxproj /p:configuration=debug /p:platform=x64
if /i {%ANSWER%}=={3} (goto :release)
goto :end

:release
echo Compile Release...
CALL "%VS120COMNTOOLS%\vsvars32.bat" x86

msbuild.exe itom\ALL_BUILD.vcxproj /p:configuration=release /p:platform=x64
msbuild.exe designerplugins\ALL_BUILD.vcxproj /p:configuration=release /p:platform=x64
msbuild.exe plugins\ALL_BUILD.vcxproj /p:configuration=release /p:platform=x64
goto :end 

:end
echo Compilation finished
pause
