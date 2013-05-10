.. include:: ../include/global.inc

Build dependencies
==========================

This setup lists third-party software packages and libraries, that are required if you want to build |itom| from sources. If you run
a setup-release of |itom| none of these dependencies (besides a python 3 installation) are required. Most of the following hints address
the build on a Windows operating system. However the required packages are mainly the same for Linux and most components can directly
be obtained by the specific package manager of your Linux distribution.

Software packages
------------------

**Required Software-Packages**

- IDE, Compiler (e.g. Visual Studio 2010 Professional, QtCreator...)
- CMake (recommended 2.8.9 or higher)
- Qt-framework (4.7 or higher)
- QScintilla2
- OpenCV 2.3 or higher
- PythonCloudLibrary 1.6 or higher
- Python 3.2 or higher
- Python-Package: NumPy

**Optional Software-Packages**

- Qt-AddOn for Visual Studio (requires .NET 2.0 framework with SP 1.0)
- Git (git-scm.com) + GUI (e.g. TortoiseGit or GitExtensions) for accessing remote repository
- Doxygen (for creating the source code documentation)
- Python-Packages: SciPy, Distribute, Sphinx (user documentation generation), scikit-image, matplotlib...

Detailed information
----------------------

**Compiler, IDE** (mandatory)

You can use any compiler and integrated developement environment (IDE) which is supported by **CMake** (http://www.cmake.org/cmake/help/v2.8.10/cmake.html#section_Generators).
On Windows systems, we develop with **Visual Studio 2010 Professional**, whereas we use **QtCreator** for the developement under Linux. QtCreator is no generator of CMake, however
QtCreator directly supports CMakeLists.txt-files. It is also possible to use the free express edition of Visual Studio.

.. note::
    
    Please consider that you need to install the Service Pack 1 for Visual Studio 2010 Professional when compiling a 64bit version of |itom|.

**CMake** (mandatory)

Download **CMake** from http://www.cmake.org/cmake/resources/software.html and install it. If possible use any version higher than 2.8.8. CMake reads the platform-independent
project files of |itom| (CMakeList.txt) and generates the corresponding project files for your compiler, IDE and platform.

**Qt-framework** (mandatory)

Download the **Qt-framework** (version 4.7 or higher, branch 5 is not supported yet) from http://qt-project.org/downloads. If you find a setup version for your IDE and compiler,
you can directly install it. Else, you need to configure and build **Qt** on your computer. This is also the case if you want to compile |itom| with Visual Studio for 64bit (see box below).

Create the following environment variables (Windows only - you need to logoff your computer in order to activate changes to environment variables):

* create an entry **QTDIR** and set it to the *Qt*-base directory (e.g. **C:\\Qt\\4.8.0**)
* create an entry **QMAKESPEC** and set it to the string **win32-msvc2010** (even if you are compiling for 64bit) or similar (see http://qt-project.org/doc/qt-4.8/qmake-environment-reference.html#qmakespec)
* add the following text to the Path variable: **;%QTDIR%\\bin** (please only **add** this string, **do not replace** the existing path-entry) 

.. note::
    
    **Compiling Qt for 64bit, Visual Studio**
    
    This side-note explains how to configure and build Qt for a 64bit build, Visual Studio 2010. The general approach for other configurations is similar.
    
    - Delete files beginning with *sync* from **%QTDIR%\\bin** directory (in order to avoid the requirement of Perl during compilation, which is not necessary in our case). 
    - 64bit: Open **Visual Studio Commandline x64 Win64 (2010)** in your Start-Menu under *Microsoft Visual Studio >> Visual Studio Tools*. 
    - (for 32bit always use the **Visual Studio Commandline (2010)**)
    - change to Qt-Dir by typing::
        
        cd %QTDIR%
      
      into the command line.
    - configure Qt-compilation by executing the command::

       configure -platform win32-msvc2010 -debug-and-release -opensource -no-qt3support -qt-sql-odbc -qt-sql-sqlite -qt-zlib -qt-libpng -webkit 
       
    - Cchoose the option **open source version** and accept the license information during the configuration process. The configuration may take between 5 and 20 minutes. 
    - now start the time-intense compilation process (1 to 5 hours) by executing the command::

       nmake
    
    If you want to restart the entire compilation you need to completely remove a possible older configuration. Then open the appropriate Visual Studio command line and
    execute::
        
        nmake distclean

**Qt-Visual Studio-AddIn** (optional, only for Visual Studio, not necessary for QtCreator)
   
If you want to have a better integration of **Qt** into **Visual Studio** (e.g. better debugging information for Qt-types like lists or vectors), you should download the
**Qt-Visual Studio-AddIn** (1.1.11 for Qt 4.8.x, 1.1.10 for Qt 4.7.x) from http://qt-project.org/downloads#qt-other and install it. Since we are using **CMake** it is not
mandatory to use this **AddIn** like it is usually the case when developing any Qt-project with Visual Studio. Therefore it is also possible to use the Express edition of
Visual Studio, where you cannot install this add-in.

**QScintilla2** (mandatory)

In general, download **QScintilla** (2.6 or higher) from http://www.riverbankcomputing.com/software/qscintilla/download and build the debug and release version. The original
QtCreator project file of QScintilla finally copies the entire output to the **bin** directory of **Qt** such that no other settings need to be set. However, the original 
project settings are not ready for a multi-configuration build in **Visual Studio**, therefore you need to tackle a little bit the Qt-project file.

An easier approach is to get the sources from **\\Obelix\\software\\m\\ITOM\\Installationen\\4. QScintilla2** (ITO only) and copy the folder **QScintilla2.6** 
to a directory on your hard drive (e.g. **C:\QScintilla2.6**, avoid Windows program directory due to restrictions in write access). 
The open your Visual Studio Command Line and change to the directory of **QScintilla** on your hard drive. 
Just execute the batch file and answer the given questions::

   qscintilla_install.bat

.. note::
    
    If the batch file breaks with some strang error messages, please make sure, that the location where **QScintilla** is installed is writable by the batch file (e.g. the
    Windows program directory under Windows Vista or higher). Therefore it is recommended to locate **QScintilla** in another directory than the program-directory.

.. _install-depend-opencv:

**OpenCV** (mandatory, 2.3 or higher)

You have different possibilities in order to get the binaries from OpenCV:

1. Download the OpenCV-Superpack (version 2.3) from http://sourceforge.net/projects/opencvlibrary/files/opencv-win/2.3/. This superpack is a self-extracting archive. Unpack it.
   This superpack contains pre-compiled binaries for VS2008, VS2010, MinGW in 32bit and 64bit. (Later map the CMake variable **OpenCV_DIR** to the **build** subdirectory of the
   extracted archive).
2. Download the current setup (version 2.4 or higher) from http://opencv.org/ and install it. This installation also contain pre-compiled binaries for VS2008, VS2010 and MinGW.
   In this case map **OpenCV_DIR** to the **opencv/build** subdirectory.
3. Get the sources from OpenCV and use CMake to generate project files and build the binaries by yourself. Then map **OpenCV_DIR** to the build-directory, indicated in CMake.

Finally, add the appropriate bin-folder of OpenCV to the windows environment variable: 
- VS2010, 32bit: ADD to the path-variable: **;C:\\OpenCV2.3\\build\\x86\\vc10\\bin** (or similar)
- VS2010, 64bit: ADD to the path-variable: **;C:\\OpenCV2.3\\build\\x64\\vc10\\bin** (or similar)

Changes to the environment variable only become active after a re-login to windows.

**PointCloudLibrary** (mandatory, 1.6 or higher)

The PointCloud-Library is a sister-project of OpenCV and is able to work with large point clouds.

The binaries can be loaded from the website http://www.pointclouds.org/downloads/windows.html. Depending on 32bit or 64bit execute the **AllInOne-Installer for Visual Studio 2010**. 
The installation directory may for example be **C:\PCL1.6.0**. Information: Please install the PCL base software including all 3rd-party packages, besides OpenNI. 
You don't have to install OpenNI, since this is only the binaries for the communication with commercial range sensors, like Kinect.

If you want to debug the point cloud library (not necessary, optional) unpack the appropriate zip-archive with the pdb-files into the bin-folder of the point cloud library. 
This is the folder where the dll's lie, too.

**Python** (mandatory, 3.2)

Download the installer from http://www.python.org/download/ and install python in version 3.2. You can simultaneously run different versions of python.

**NumPy** (mandatory)

Get a version of NumPy that fits to python 3.2 and install it. On windows binaries for many python packages can be found under http://www.lfd.uci.edu/~gohlke/pythonlibs/.

**Sphinx** (optional)

The Python package **Sphinx** is used for generating the user documentation of |itom|. You can also download sphinx from http://www.lfd.uci.edu/~gohlke/pythonlibs/. However,
sphinx is dependent on other packages, such that it is worth to get the package **distribute** (see below) first. Then open a command-line (cmd.exe) and move to the directory **[YourPythonPath]/Scripts**.
Type the following command in order to download **sphinx** including dependecies from the internet and install it::
    
    easy_install -U sphinx

**Distribute** (optional)

Distribute is a python package, than can be used for easily downloading and installing other python packages, like *Sphinx*. Download the latest version (file ending with tar.gz) of
**Distribute** from https://pypi.python.org/pypi/distribute and unpack it to any temporary directory on your hard drive. Open a command line and switch to the directory of **Distribute**.

Assuming that Python is located under **C:\Python32**, execute the following command::
    
    C:\python32\python.exe setup.py install

The **Distribute** is installed and you can use the **easy_install** tool (see **Sphinx** installation above).

**Other python packages** (optional)

You can always check the website http://www.lfd.uci.edu/~gohlke/pythonlibs/ for appropriate binaries of your desired python package.

.. note::
    
    If you use any python packages depending on NumPy (e.g. SciPy, scikit-image...) try to have corresponding versions. If your SciPy installation is younger than NumPy,
    some methods can not be executed and a python error message is raised, saying that you should update your NumPy installation.