@echo off

SET /P ANSWER=designerplugins[1], plugins[2], itom[3], all[4]?
echo You chose: %ANSWER%
if /i {%ANSWER%}=={1} (goto :designerplugins)
if /i {%ANSWER%}=={2} (goto :plugins)
if /i {%ANSWER%}=={3} (goto :itom)
if /i {%ANSWER%}=={4} (goto :designerplugins)


echo Wrong input.
goto :end

:designerplugins

echo fetch designerplugins
cd designerplugins
git.exe fetch --progress -v "origin" master:remotes/origin/master
cd..

if /i {%ANSWER%}=={4} (goto :plugins)
goto :end

:plugins

echo fetch plugins
cd plugins
git.exe fetch --progress -v "origin" master:remotes/origin/master
cd..

if /i {%ANSWER%}=={4} (goto :itom)
goto :end

:itom

echo fetch itom
cd itom
git.exe fetch --progress -v "origin" master:remotes/origin/master
cd..

goto :end

:end
echo Git fetch finished
pause
