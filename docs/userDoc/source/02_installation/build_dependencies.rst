.. include:: ../include/global.inc

Build dependencies
==========================

This setup lists third-party software packages and libraries that are required
if you want to build |itom| from sources. If you run a setup-release of |itom|
none of these dependencies (besides a python 3 installation) are required.
Most of the following hints address the build on a Windows operating system.
However the required packages are mainly the same for Linux and most components can
directly be obtained by the specific package manager of your Linux distribution.

Software packages
------------------

**Required Software-Packages**

- IDE (e.g. Visual Studio 2022 Professional, QtCreator...)
- Compiler: The C++ compiler must support at least the **C++11** standard.
- CMake (recommended 3.12 or higher)
- Qt5-framework or Qt6-framework (>= 5.5 required, >= 5.6 recommended)
- OpenCV 3.2 or higher (4.x recommended)
- Python 3.5 or higher, 3.7 or higher recommended
- Git (git-scm.com) + GUI (e.g. TortoiseGit or GitExtensions) for accessing the remote repository
- Python-Package: Numpy (up to itom 4.3.0, Numpy < 2.0 is supported only, Numpy 2.0 support will be added afterwards)

**Optional Software-Packages**

- PointCloudLibrary 1.6 or higher (>= 1.9 recommended, optional)
- Qt-AddOn for Visual Studio (requires .NET 2.0 framework with SP 1.0)
- Doxygen (for creating the source code documentation)
- Python-Packages: Scipy, Sphinx + numpydoc + breathe (user documentation generation), scikit-image, matplotlib...

Detailed information
----------------------

**Compiler, IDE** (mandatory)
''''''''''''''''''''''''''''''''

You can use any compiler and integrated development environment (IDE) which is
supported by **CMake** (https://cmake.org/cmake/help/latest/manual/cmake-generators.7.html).
On Windows systems, we develop with **Visual Studio 2015+ Professional**, whereas
we use **QtCreator** for the development under Linux. QtCreator is no specific
CMake generator, however QtCreator directly supports CMakeLists.txt-files.
It is also possible to use the free express edition of Visual Studio.

.. note::

    Please consider that you need to install the Service Pack 1 if you want to
    use Visual Studio 2015 Professional to compile a 64bit version of |itom|.
    It is even recommended to install the service
    pack for a 32bit compilation.

.. note::

    The C++ compiler must support at least the **C++11** standard.


**CMake** (mandatory)
''''''''''''''''''''''''''''''''

Download **CMake** from http://www.cmake.org/cmake/resources/software.html and install it.
Or just download it and use that. no need to install.
If possible use any version higher than 3.12. CMake reads the platform-independent
project files of |itom| (CMakeList.txt) and generates the corresponding project
files for your compiler, IDE and platform.

**Qt-framework** (mandatory)
''''''''''''''''''''''''''''''''

Download the **Qt5-framework or Qt6-framework** (>= 5.5 required, >= 5.6 recommended)
from http://qt-project.org/downloads. If you find a setup version for your IDE and compiler,
you can directly install it. Otherwise, you need to configure and build **Qt**
on your computer - see box below. Either download the ready-to-use binaries from
qt-project.org, compile it from sources and follow the instruction in the box
below or consider to use the :ref:`all-in-one-development setup <install-all-in-one-dev-setup>`).
If you use the ready-to-use binaries, make sure to use a version with OpenGL.

Create the following environment variables (Windows only - you need to log-off
from your computer in order to activate changes to environment variables):

* create an entry **QTDIR** and set it to the *Qt*-base directory (e.g. **C:\\Qt\\5.12.0**)
* Create an entry **QMAKESPEC** in System Environment Variables and set it to the
    string **win32-msvc2010**. For MSVC 2017 or newer **QMAKESPEC** must be set to **win32-msvc**.
* add the following text to the Path variable: **;%QTDIR%\\bin**
    (please only **add** this string, **do not replace** the existing path-entry)

.. _compile-qt5-from-sources:

.. note::

    **Compiling Qt 5.x (using the example 64bit, Visual Studio 2015)**

    This side-note explains how to configure and build Qt5 for a 64bit build using Visual Studio 2010.
    The general approach for other configurations is similar.

    - Download the Qt5 sources a zip or tar.gz archive from qt-project.org and unpack them (e.g. to C:\Qt5.6.0)
    - Make sure that Python 3.x is installed and verify that the path, containing
        the application **python.exe** is contained in the Windows environment variable.
    - Open **Visual Studio Commandline x64 Win64 (2010)** (64bit) or
        **Visual Studio Commandline (2010)** (32bit) e.g.
        via *Windows >> Start >> Microsoft Visual Studio >> Visual Studio Tools*.
    - Change to Qt-directory by typing::

        cd %QTDIR%

    - configure Qt by executing the command::

        configure -platform win32-msvc2010 -debug-and-release -opensource -nomake tests -nomake examples -qt-sql-odbc -qt-sql-sqlite -qt-zlib -qt-libpng -opengl desktop -skip qtwebkit -no-icu -no-openssl -skip qtscript -skip qtquick1

    - choose the option **open source version** and accept the license information.
    - now start the time-intense compilation process by executing::

        nmake

    - If ready type::

        nmake install

    - Finally, you can build the documentation (build into the qtbase/doc folder)
        by typing (see http://qt-project.org/wiki/Building_Qt_Documentation)::

            nmake docs

    If you want to restart the entire compilation you need to completely remove
    any possible older configuration. Then open the appropriate Visual Studio command line and
    execute::

        nmake distclean

    If Python could not be accessed, an error during the compilation may occur.
    Then make sure that Python is accessible via the Path environment variable and delete
    the possibly available file *C:/Qt/Qt5.3.0/qtdeclarative/src/qml/RegExpJitTables.h*.

    :ref:`build_dependencies_qt`


**Qt-Visual Studio-AddIn** (optional, only for Visual Studio, not necessary for QtCreator)
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

If you want to have a better integration of **Qt** into **Visual Studio**
(e.g. better debugging information for Qt-types like lists or vectors),
you should download the **Qt-Visual Studio-AddIn** (1.2.x for Qt 5.x,) from
http://qt-project.org/downloads#qt-other or via visual studio gallery and install it.
Since we are using **CMake** it is not mandatory to use this **AddIn** like
it is usually the case when developing any Qt-project with Visual Studio.
Therefore it is also possible to use the Express edition of
Visual Studio, where you cannot install this add-in. The **Qt Visual Studio AddIn**
requires that you have the **.NET framework 2.0 SP 1** installed on your PC.

.. note::

    Sometimes, there are problems when starting Visual Studio with an installed
    Qt-AddIn. In case that any component cannot be registered, as warned by a message-box when
    starting Visual Studio, you should check the bug and its fix described at
    https://bugreports.qt-project.org/browse/QTVSADDINBUG-77. In most cases it was sufficient to register
    the library **stdole.dll** using the tool **gacutil.exe** from the *
    *Microsoft SDKs/Windows/v7.0A/bin** subfolder of your standard program folder.
    Start a windows commandline and move to the directory on your computer
    where the executable program *gacutil.exe* is located, then type::

        gacutil.exe -i "C:\Program Files (x86)\Common Files\microsoft shared\MSEnv\PublicAssemblies\stdole.dll"

.. note::

    Visual Studio Community Edition

    You can also use the Community Editions of Visual Studio (e.g. Visual Studo 2015 Community Editions) to compile itom.

.. _install-depend-opencv:

**OpenCV** (mandatory, 2.3 or higher, 3.x recommended, 4.x is also possible)
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

You have different possibilities in order to get the binaries from OpenCV:

1. Download the OpenCV-Superpack (version 2.3) from http://sourceforge.net/projects/opencvlibrary/files/opencv-win/2.3/.
   This superpack is a self-extracting archive. Unpack it.
   The superpack contains pre-compiled binaries for VS2008, VS2010, MinGW in 32bit and 64bit.
   (Later map the CMake variable **OpenCV_DIR** to the **build** subdirectory of the
   extracted archive).
2. Download the current setup (version 2.4 or higher recommended) from http://opencv.org/
   and install it. This installation also contains pre-compiled binaries for VS2008, VS2010 and MinGW.
   In this case map **OpenCV_DIR** to the **opencv/build** subdirectory.
3. Get the sources from OpenCV and use CMake to generate project files and build the binaries by yourself.
   Then map **OpenCV_DIR** to the build-directory, indicated in CMake.

Finally, add the appropriate bin-folder of OpenCV to the windows environment variable:
- VS2010, 32bit: Add to the path-variable: **;C:\\OpenCV2.3\\build\\x86\\vc10\\bin** (or similar)
- VS2010, 64bit: Add to the path-variable: **;C:\\OpenCV2.3\\build\\x64\\vc10\\bin** (or similar)

Changes to the environment variable only become active after a re-login to windows.

.. note::

    There is a known linker problem with OpenCV 2.4.7 (only this version). Please avoid to use this special version.

:ref:`build_dependencies_opencv`


**PointCloudLibrary** (optional, >= 1.6, 1.9 or higher recommended)
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

The PointCloud-Library is a sister-project of OpenCV and is able to work with
large point clouds. You can compile |itom| with support for the point cloud library.
Then the python classes **itom.pointCloud**, **itom.point** and **itom.polygonMesh**
are available and algorithm plugins can use point cloud functionalities. If you
don't need anything like this, don't install the point cloud library and uncheck
the option **BUILD_WITH_PCL** in the CMake configurations of |itom|.

The binaries can be loaded from the website http://www.pointclouds.org/downloads/windows.html.
Depending on 32bit or 64bit execute the **AllInOne-Installer for Visual Studio 201x**.
The installation directory may for example be **C:\\PCL1.9.1**.
Information: Please install the PCL base software including all 3rd-party packages,
besides OpenNI. You don't have to install OpenNI, since this is only the binaries
for the communication with commercial range sensors, like Kinect.

If you want to debug the point cloud library (not necessary, optional) unpack the
appropriate zip-archive with the pdb-files into the bin-folder of the point cloud library.
This is the folder where the dll's are located as well.

Add the path to the bin-folder of PointCloud-library to the windows environment variable:

- Add to the path-variable: **;C:\\PCL\\1.9.1\\bin** (or similar)

:ref:`build_dependencies_pcl`


**VTK** (optional, better know what you're doing...)
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

:ref:`build_dependencies_vtk`

**Python** (mandatory, 3.5 or higher, 3.7 or higher is recommended)
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

Download the installer from http://www.python.org/download/ and install python in
version 3.5 or higher. You can simultaneously run different versions of python.

**NumPy** (mandatory)
''''''''''''''''''''''''''

Get a version of NumPy that fits to your python version and install it.
On Windows, binaries for many python packages can be found under
http://www.lfd.uci.edu/~gohlke/pythonlibs/. But you can also find them more easily
on pypi and you can then install them using pip directly.

**pip** (optional)
'''''''''''''''''''''

**Pip** is the new package installation tool for |python| packages. If you don't
have **pip** already installed (already included in Python >= 3.4) use the following
hints to get **pip**. Download the file from https://raw.github.com/pypa/pip/master/contrib/get-pip.py
and save it to any temporary directory. Then open the file **get-pip.py** with
the python version used for compiling |itom| (e.g. python32.exe). As an alternative,
open a command line and switch to the directory where you save the file **get-pip.py**.

Assuming that Python is located under **C:\\Python32**, execute the following command::

    C:\\python32\\python.exe get-pip.py

**pip** is installed and you can use the **pip** tool (see **Sphinx** installation above).

**Sphinx** (optional)
''''''''''''''''''''''''''''

The Python package **Sphinx** is used for generating the user documentation of |itom|.
You can also download sphinx from http://www.lfd.uci.edu/~gohlke/pythonlibs/. However,
sphinx is dependent on other packages, such that it is worth to install Sphinx
using the |python| tool **pip** (If you don't have **pip** see the next section).
Then open a command-line (cmd.exe) and switch to the directory **[YourPythonPath]/Scripts**.
Type the following command in order to download **sphinx** including dependencies
from the internet and install it::

    pip install sphinx

For upgrading **sphinx**, type::

    pip install sphinx --upgrade

Next to **Sphinx** also install the **numpydoc** package if you want to build
the user documentation.

**jedi** (optional)
'''''''''''''''''''''

For auto completion, calltips, goto definition features etc. install the
Python package **jedi** using pip::

    pip install jedi

**flake8** (optional)
''''''''''''''''''''''

**flake8** provides extended code checker functionalities, that is
integrated into the GUI of itom. **flake8** combines the code checker
functionalities of the packages **pyflakes**, **pycodestyle** and **mccabe**.
It can further be extended by other packages / plugins. Install **flake8**
via pip::

    pip install flake8

**pyflakes** (optional)
'''''''''''''''''''''''''''

Instead of **flake8**, it is also possible to only install **pyflakes**
for a reduced set of code checker functionality. If installed, your scripts
are automatically checked for syntax errors that are marked
as bug symbols in each line. The detailed messages are displayed as tool tip texts
of the bug symbol. Use pip to install this package::

    pip install pyflakes


**Other python packages** (optional)
'''''''''''''''''''''''''''''''''''''''''

You can always check the website http://www.lfd.uci.edu/~gohlke/pythonlibs/ for
appropriate binaries of your desired python package.

.. note::

    If you use any python packages depending on NumPy (e.g. SciPy, scikit-image...)
    try to have corresponding versions. If your SciPy installation is younger than NumPy,
    some methods can not be executed and a python error message is raised, saying
    that you should update your NumPy installation.
