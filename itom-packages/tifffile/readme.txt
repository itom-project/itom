http://www.lfd.uci.edu/~gohlke/ >> Transformations.py and transformations.c


You need to compile transformations.c by python.

VS 2010
-------

1. Open MS Visual Studio 2010 command line (32bit or 64bit, depending on itom and python)
2. goto this folder and type
    C:\python32\python.exe setup.py build_ext --inplace

3. If the compilation exits with an error, python cannot find the VS2010 compiler. Therefore
type in the same commandline
   SET VS90COMNTOOLS=%VS100COMNTOOLS%

and repeat step 2


Hints about problem point 3
---------------------------

http://magic-smoke.blogspot.de/2012/07/building-pyliblo-on-windows-using.html
SECTION "Build pyliblo"
