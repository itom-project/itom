All-In-One Development Environments
=====================================

This folder contain zip-archives with all required tools and libraries to compile itom
for various development environments.

More information how to get your computer ready to compile itom and its plugins
using these all-in-one development environments can be found here: 
http://itom.bitbucket.org/latest/docs/02_installation/install_all-in-one_development_setup.html

Available editions
-----------------------------

**MSVC2010-x86 / MSVC2010-x64**

* 32bit / 64bit
* Visual Studio 2010
* Qt 5.3.2
* Python 3.4.2 / Numpy 1.8.2
* CMake 3.0.2
* OpenCV 2.4.10

**MSVC2013-x64**

* 64bit
* Visual Studio 2013 (e.g. Community Edition)
* Qt 5.4.1
* Python 3.4.2 / Numpy 1.8.2
* CMake 3.2.2
* OpenCV 2.4.11

**MSVC2010-Qt5.6.0-x86 / MSVC2010-Qt5.6.0-x64**

* 32bit / 64bit
* Visual Studio 2013 (e.g. Community Edition)
* Qt 5.6.0
* Python 3.5.1 / Numpy 1.11.0
* CMake 3.5.1
* OpenCV 3.1.0

Changelog
--------------------

2016-04-27
~~~~~~~~~~~~~~~~~~
All-In-One Development Environment for Visual Studio 2013 (tested with free Community version) created for x86 and x64 and the new Qt 5.6.0.

It comes with:
Git 2.8.1
Python 3.5.1
Numpy MKL 1.11.0
Qt 5.6.0
OpenCV 3.1.0
CMake 3.5.1
QScintilla 2.9.1

and optional:

Boost 1.60.0
Eigen 3.2.7
Flann 1.7.1
QHull 2015.2
VTK 7.0.0
PCL 1.8.0

2015-11-18
~~~~~~~~~~~~~~~~~~
Bugfix in patch file list to patch Qt5 installation in VS2013_x64 version.

2015-09-17
~~~~~~~~~~~~~~~~~~
CMake caused an error if build with PCL was enabled. This error said that the variable Boost_LIBRARY_DIR is invalid.
It has to be replaced by BOOST_LIBRARY_DIR. This was fixed in all setup.py files in all configurations.

2015-05-05
~~~~~~~~~~~~~~~~~~
All-In-One Development Environment for Visual Studio 2013 (tested with free Community version) created for x64.

It comes with:
Python 3.4.2
Git 1.9.5
Numpy 1.8.2
Cmake 3.2.2
Qt5.4.1 with QtWebkit
OpenCV 2.4.11
QScintilla 2.9

and optional:

PCL 1.8.0
VTK 6.1.0
Boost 1.58
Eigen 3.0.5
Flann 1.7.1
QHull 2012.1


2015-04-29
~~~~~~~~~~~~~~~~~~
Added Boost_LIBRARY_DIR to CMake to enable a more robust search for boost.

2015-04-01
~~~~~~~~~~~~~~~~~~
CONSIDER_GIT_SVN in CMake of itom is set to True in order to check the git version numbers. 
PAUSE command after each build is removed such that an automatic run without user interaction using option 7 is possible.

2015-03-19
~~~~~~~~~~~~~~~~~~
Bugfix if the path to the all-in-one installation contains spaces

2015-02-20
~~~~~~~~~~~~~~~~~~
A syntax error has been fixed in the 64bit setup.py Python script (missing indentation in if-case)

2015-01-16
~~~~~~~~~~~~~~~~~~
A bug is fixed in the setup.py Python script, such that the source directory is automatically created if it doesn't exist yet.
With the old version, the clone process of git for itom, the plugins or designer-plugins might fail if the source directory, as 
parent of the specific sub-directories, didn't exist.

2015-01-02
~~~~~~~~~~~~~~~~~~
The all-in-one installers are updated concerning Qt. Qt is now precompiled with QtWebkit for allowing a proper appearance
of help contents in the assistant. It also comes with the Qt help pages. Qt with the webkit tool also requires the ICU
project that is also contained in the base package.

2014-12-01
~~~~~~~~~~~~~~~~~~
The new all-in-one installers for a VS2010 development environment for compiling itom and its plugins has been released.
It comes with:

Python 3.4.2
Git 1.9.4
Numpy 1.8.2
Cmake 3.0.2
Qt5.3.2 with QtWebkit
OpenCV 2.4.10
QScintilla 2.8

and optional:

PCL 1.8.0
VTK 6.1.0
Boost 1.57
Eigen 3.0.5
Flann 1.7.1
QHull 2011.1


Known Problems
~~~~~~~~~~~~~~~~~~
If the Visual Studio Qt AddIn does not start (error: couldn't register all qt4vsaddin commands...), try to open the 
Microsoft Visual Studio command line (x86 or x64 depending on your build) and run:

gacutil.exe -i "C:\Program Files (x86)\Common Files\microsoft shared\MSEnv\PublicAssemblies\stdole.dll"

See: https://bugreports.qt.io/browse/QTVSADDINBUG-77 for more information
