.. include:: ../include/global.inc

Build dependencies
==========================

This setup lists third-party software packages and libraries that are required if you want to build |itom| from sources. If you run
a setup-release of |itom| none of these dependencies (besides a python 3 installation) are required. Most of the following hints address
the build on a Windows operating system. However the required packages are mainly the same for Linux and most components can directly
be obtained by the specific package manager of your Linux distribution.

Software packages
------------------

**Required Software-Packages**

- IDE, Compiler (e.g. Visual Studio 2010 Professional, QtCreator...)
- CMake (recommended 2.8.9 or higher)
- Qt-framework (4.7 or higher, 4.8 recommended, 5.x in preparation)
- QScintilla2 (2.6 or higher)
- OpenCV 2.3 or higher (2.4 recommended)
- Python 3.2 or higher
- Git (git-scm.com) + GUI (e.g. TortoiseGit or GitExtensions) for accessing the remote repository
- Python-Package: NumPy

**Optional Software-Packages**

- PointCloudLibrary 1.6 or higher (optional)
- Qt-AddOn for Visual Studio (requires .NET 2.0 framework with SP 1.0)
- Doxygen (for creating the source code documentation)
- Python-Packages: SciPy, Distribute, Sphinx (user documentation generation), scikit-image, matplotlib...

Detailed information
----------------------

**Compiler, IDE** (mandatory)

You can use any compiler and integrated development environment (IDE) which is supported by **CMake** (http://www.cmake.org/cmake/help/v2.8.10/cmake.html#section_Generators).
On Windows systems, we develop with **Visual Studio 2010 Professional**, whereas we use **QtCreator** for the development under Linux. QtCreator is no specific CMake generator, however
QtCreator directly supports CMakeLists.txt-files. It is also possible to use the free express edition of Visual Studio.

.. note::
    
    Please consider that you need to install the Service Pack 1 for Visual Studio 2010 Professional when compiling a 64bit version of |itom|. It is even recommended to install the service
    pack for a 32bit compilation.

**CMake** (mandatory)

Download **CMake** from http://www.cmake.org/cmake/resources/software.html and install it. If possible use any version higher than 2.8.8. CMake reads the platform-independent project files of |itom| (CMakeList.txt) and generates the corresponding project files for your compiler, IDE and platform.

**Qt-framework** (mandatory)

Download the **Qt-framework** (version 4.7 or higher, 4.8.x recommended, 5.x is coming soon) from http://qt-project.org/downloads. If you find a setup version for your IDE and compiler, 
you can directly install it. Otherwise, you need to configure and build **Qt** on your computer - see box below (e.g. Qt 4.8.x with Visual Studio 2010, 64bit needs to be compiled). For
Qt5 either download the ready-to-use binaries from qt-project.org or compile it from sources and follow the instruction in the box below). If you use the ready-to-use binaries, make sure
to use a version with OpenGL.

Create the following environment variables (Windows only - you need to log-off from your computer in order to activate changes to environment variables):

* create an entry **QTDIR** and set it to the *Qt*-base directory (e.g. **C:\\Qt\\4.8.0**)
* create an entry **QMAKESPEC** and set it to the string **win32-msvc2010** (even if you are compiling for 64bit) or similar (see http://qt-project.org/doc/qt-4.8/qmake-environment-reference.html#qmakespec)
* add the following text to the Path variable: **;%QTDIR%\\bin** (please only **add** this string, **do not replace** the existing path-entry)

.. note::
    
    **Compiling Qt 4.7.x or 4.8.x (using the example 64bit, Visual Studio)**
    
    This side-note explains how to configure and build Qt for a 64bit build using Visual Studio 2010. The general approach for other configurations is similar.
    
    - Delete files beginning with *sync* from the **%QTDIR%\\bin** directory (in order to avoid the requirement of Perl during compilation, which is not necessary in our case). 
    - 64bit: Open **Visual Studio Commandline x64 Win64 (2010)** in your Start-Menu under *Microsoft Visual Studio >> Visual Studio Tools*. 
    - (for 32bit always use the **Visual Studio Commandline (2010)**)
    - change to Qt-directory by typing::
        
        cd %QTDIR%
      
      into the command line.
    - configure Qt-compilation by executing the command::

       configure -platform win32-msvc2010 -debug-and-release -opensource -no-qt3support -qt-sql-odbc -qt-sql-sqlite -qt-zlib -qt-libpng -webkit 
       
    - choose the option **open source version** and accept the license information during the configuration process. The configuration may take between 5 and 20 minutes. 
    - now start the time-intense compilation process (1 to 5 hours) by executing the command::

       nmake
    
    If you want to restart the entire compilation you need to completely remove any possible older configuration. Then open the appropriate Visual Studio command line and
    execute::
        
        nmake distclean
        
.. _compile-qt5-from-sources:
        
.. note::
    
    **Compiling Qt 5.x (using the example 64bit, Visual Studio 2010)**
    
    This side-note explains how to configure and build Qt5 for a 64bit build using Visual Studio 2010. The general approach for other configurations is similar.
    
    - Download the Qt5 sources a zip or tar.gz archive from qt-project.org and unpack them (e.g. to C:\Qt5.3.0)
    - Make sure that Python 3.x is installed and verify that the path, containing the application **python.exe** is contained in the Windows environment variable.
    - Open **Visual Studio Commandline x64 Win64 (2010)** (64bit) or **Visual Studio Commandline (2010)** (32bit) e.g. via *Windows >> Start >> Microsoft Visual Studio >> Visual Studio Tools*.
    - Change to Qt-directory by typing::
        
        cd %QTDIR%
    
    - configure Qt by executing the command::
        
        configure -platform win32-msvc2010 -debug-and-release -opensource -nomake tests -nomake examples -qt-sql-odbc -qt-sql-sqlite -qt-zlib -qt-libpng -opengl desktop -skip qtwebkit -no-icu -no-openssl -skip qtscript -skip qtquick1
    
    - choose the option **open source version** and accept the license information.
    - now start the time-intense compilation process by executing::
        
        nmake
    
    - If ready type::
        
        nmake install
        
    If you want to restart the entire compilation you need to completely remove any possible older configuration. Then open the appropriate Visual Studio command line and
    execute::
        
        nmake distclean
    
    If Python could not be accessed, an error during the compilation may occur. Then make sure that Python is accessible via the Path environment variable and delete
    the possibly available file *C:/Qt/Qt5.3.0/qtdeclarative/src/qml/RegExpJitTables.h*.
    
    

**Qt-Visual Studio-AddIn** (optional, only for Visual Studio, not necessary for QtCreator)
   
If you want to have a better integration of **Qt** into **Visual Studio** (e.g. better debugging information for Qt-types like lists or vectors), you should download the
**Qt-Visual Studio-AddIn** (1.2.x for Qt 5.x, 1.1.11 for Qt 4.8.x, 1.1.10 for Qt 4.7.x) from http://qt-project.org/downloads#qt-other and install it. Since we are using **CMake** it is not
mandatory to use this **AddIn** like it is usually the case when developing any Qt-project with Visual Studio. Therefore it is also possible to use the Express edition of
Visual Studio, where you cannot install this add-in. The **Qt Visual Studio AddIn** requires that you have the **.NET framework 2.0 SP 1** installed on your PC.

.. note::
    
    Sometimes, there are problems when starting Visual Studio with an installed Qt-AddIn. In case that any component cannot be registered, as warned by a message-box when
    starting Visual Studio, you should check the bug and its fix described at https://bugreports.qt-project.org/browse/QTVSADDINBUG-77. In most cases it was sufficient to register
    the library **stdole.dll** using the tool **gacutil.exe** from the **Microsoft SDKs/Windows/v7.0A/bin** subfolder of your standard program folder. Start a windows commandline and move to the directory on your computer
    where the executable program *gacutil.exe* is located, then type::
        
        gacutil.exe -i "C:\Program Files (x86)\Common Files\microsoft shared\MSEnv\PublicAssemblies\stdole.dll"

**QScintilla2** (mandatory)

Download **QScintilla** (2.6 or higher) from http://www.riverbankcomputing.com/software/qscintilla/download and build the debug and release version. The original
QtCreator project file of QScintilla finally copies the entire output to the **bin** directory of **Qt** so that no other settings need to be adapted. However, the original 
project settings are not ready for a multi-configuration build in **Visual Studio**. As a result, you need to adapt the Qt-project file. To do this follow these steps:

    * Copy the downloaded files to a directory of your choise (preferably **NOT** the windows program directory, we are assuming in the following that you placed them in **C:\\QScintilla2**)
    * Open the Visual-Studio 2010 32-bit commandline (or 64-bit if installed)
    * Open the file **C:\\QScintilla2\\Qt4\\QScintilla.pro** or **C:\\QScintilla2\\Qt4Qt5\\QScintilla.pro** (version 2.8 or higher) in a text editor and replace the line **CONFIG** with::
        
        CONFIG += qt warn_off debug_and_release build_all dll thread
    
    and add the lines::
        
        CONFIG(debug, debug|release){ TARGET = $$join(TARGET,,,d) }
    
    (see also: http://www.mantidproject.org/Debugging_MantidPlot)
    
    * If you had a previous installation of QScintilla, delete the directory **%QTDIR%\\include\\Qsci** as well as the files called **qscintilla2.dll** and **qscintilla2d.dll** in the directory **%QTDIR%\\bin**
    * Execute the following commands from the command-line::
        
        - cd C:\\QScintilla2\\Qt4
        - nmake distclean 
        - %QTDIR%\\bin\\qmake qscintilla.pro spec=win32-msvc2010
        - nmake
        - nmake install
    
    * copy the library files qscintilla2.dll and qscintilla2d.dll from %QTDIR%\\lib to %QTDIR%\\bin 

.. note::
    
    This is for ITO only (internal use): An easier approach is to get the sources from **\\Obelix\\software\\m\\ITOM\\Installationen\\4. QScintilla2** and copy the folder **QScintilla2.8** to a directory on your hard drive (e.g. **C:\QScintilla2.8**, avoid Windows program directory due to restrictions in write access). Open your Visual Studio Command Line and change to the directory of **QScintilla** on your hard drive. Just execute the batch file **qscintilla_install.bat** and answer the given questions.

.. note::
    
    If the batch file breaks with some strange error messages, please make sure, that the location where **QScintilla** is installed is writable by the batch file (e.g. the
    Windows program directory under Windows Vista or higher). Therefore it is recommended to locate **QScintilla** in another directory than the program-directory.

.. _install-depend-opencv:

**OpenCV** (mandatory, 2.3 or higher, 2.4.x recommended)

You have different possibilities in order to get the binaries from OpenCV:

1. Download the OpenCV-Superpack (version 2.3) from http://sourceforge.net/projects/opencvlibrary/files/opencv-win/2.3/. This superpack is a self-extracting archive. Unpack it.
   The superpack contains pre-compiled binaries for VS2008, VS2010, MinGW in 32bit and 64bit. (Later map the CMake variable **OpenCV_DIR** to the **build** subdirectory of the
   extracted archive).
2. Download the current setup (version 2.4 or higher, recommended) from http://opencv.org/ and install it. This installation also contains pre-compiled binaries for VS2008, VS2010 and MinGW.
   In this case map **OpenCV_DIR** to the **opencv/build** subdirectory.
3. Get the sources from OpenCV and use CMake to generate project files and build the binaries by yourself. Then map **OpenCV_DIR** to the build-directory, indicated in CMake.

Finally, add the appropriate bin-folder of OpenCV to the windows environment variable: 
- VS2010, 32bit: Add to the path-variable: **;C:\\OpenCV2.3\\build\\x86\\vc10\\bin** (or similar)
- VS2010, 64bit: Add to the path-variable: **;C:\\OpenCV2.3\\build\\x64\\vc10\\bin** (or similar)

Changes to the environment variable only become active after a re-login to windows.

.. note::
    
    There is a known linker problem with OpenCV 2.4.7 (only this version). Please avoid to use this special version.

**PointCloudLibrary** (optional, 1.6 or higher)

The PointCloud-Library is a sister-project of OpenCV and is able to work with large point clouds. You can compile |itom| with support for the point cloud library. Then the python classes **itom.pointCloud**, **itom.point** and **itom.polygonMesh** are available and algorithm plugins can use point cloud functionalities. If you don't need anything like this, don't install the point cloud library and uncheck the option **BUILD_WITH_PCL** in the CMake configurations of |itom|.

The binaries can be loaded from the website http://www.pointclouds.org/downloads/windows.html. Depending on 32bit or 64bit execute the **AllInOne-Installer for Visual Studio 2010**. 
The installation directory may for example be **C:\\PCL1.6.0**. Information: Please install the PCL base software including all 3rd-party packages, besides OpenNI. You don't have to install OpenNI, since this is only the binaries for the communication with commercial range sensors, like Kinect.

If you want to debug the point cloud library (not necessary, optional) unpack the appropriate zip-archive with the pdb-files into the bin-folder of the point cloud library. 
This is the folder where the dll's are located as well.

Add the path to the bin-folder of PointCloud-library to the windows environment variable:

- Add to the path-variable: **;C:\\PCL\\1.6.0\\bin** (or similar)

**Python** (mandatory, 3.2 or higher)

Download the installer from http://www.python.org/download/ and install python in version 3.2 or higher. You can simultaneously run different versions of python.

**NumPy** (mandatory)

Get a version of NumPy that fits to your python version and install it. On Windows, binaries for many python packages can be found under http://www.lfd.uci.edu/~gohlke/pythonlibs/.

**pip** (optional)

**Pip** is the new package installation tool for |python| packages. If you don't have **pip** already installed use the following hints to get **pip**. Download the file from https://raw.github.com/pypa/pip/master/contrib/get-pip.py and save it to any temporary directory. Then open the file **get-pip.py** with the python version used for compiling |itom| (e.g. python32.exe). As an alternative, open a command line and switch to the directory where you save the file **get-pip.py**.

Assuming that Python is located under **C:\\Python32**, execute the following command::
    
    C:\\python32\\python.exe get-pip.py

**pip** is installed and you can use the **pip** tool (see **Sphinx** installation above).

**Sphinx** (optional)

The Python package **Sphinx** is used for generating the user documentation of |itom|. You can also download sphinx from http://www.lfd.uci.edu/~gohlke/pythonlibs/. However,
sphinx is dependent on other packages, such that it is worth to install Sphinx using the |python| tool **pip** (If you don't have **pip** see the next section). Then open a command-line (cmd.exe) and switch to the directory **[YourPythonPath]/Scripts**. Type the following command in order to download **sphinx** including dependecies from the internet and install it::
    
    pip install sphinx

For upgrading **sphinx**, type::
    
    pip install sphinx --upgrade
    
**frosted** (optional)

The Python package **frosted** can be installed in order to enable a code syntax checker in |itom|. If installed, your scripts are automatically checked for syntax errors that are marked
as bug symbols in each line. The detailed messages are displayed as tool tip texts of the bug symbol. Use pip to install this package::
    
    pip install frosted


**Other python packages** (optional)

You can always check the website http://www.lfd.uci.edu/~gohlke/pythonlibs/ for appropriate binaries of your desired python package.

.. note::
    
    If you use any python packages depending on NumPy (e.g. SciPy, scikit-image...) try to have corresponding versions. If your SciPy installation is younger than NumPy,
    some methods can not be executed and a python error message is raised, saying that you should update your NumPy installation.
