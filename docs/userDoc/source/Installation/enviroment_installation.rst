.. include:: ../include/global.inc

Component-by-component Installation
======================================

Required Software-Packages
---------------------------

* Visual Studio 2010 Professional 
* latest Version of Qt (4.8 or higher)
* Qt-AddOn for Visual Studio (requires .NET 2.0 framework with SP 1.0)
* QScintilla2
* Python 3.2 (e.g. 3.2.2)
* Python-Packages: NumPy (mandatory), SciPy, MatplotLib, ... (optionally)
* OpenCV 2.3 or 2.4 (do not use version 2.2) 

**Further software which is useful:**

* **TortoiseSVN** for accessing the trunk from the internal SVN-system
* **Doxygen** for creating the source code documentation
* **Python-Packages:** distribute and Sphinx (for user documentation generation) 


32bit or 64bit-Versions
-----------------------

* If you are running a 32bit computer, this decision is already done. You will compile the whole solution as well as Qt,... for a 32bit-system. 
* On a 64bit computer, you can decide whether you want to compile itom as 32bit or 64bit version. Unfortunately it is more difficult to prepare the same computer for both. Since we started devoloping the 32bit version, this might be recommended for "beginners". Please consider, that you also must install the corresponding python versions. The software on the Obelix server contains only 32bit versions. 
* During the installation process, you will need the Visual Studio Command Line (Kommandozeileneditor from Visual Studio) in order to start some compiling steps using the Visual Studio Compiler nmake. This command line can be found in your windows start menu. Navigate to Microsoft Visual Studio >> Visual Studio Tools. If you will compile for a 32bit version, always use the command line called Microsoft Visual Studio Commandline (2010). For compiling Qt,... with respect to a 64bit version use Microsoft Visual Studio Commandline x64 Win64 (2010). 


Introductionary Hints
---------------------

#. Please consider that the whole installation process needs some hours. The longest step (compilation of Qt for Visual Studio 2010) can be run during the night. 
#. If you use a laptop, please make sure, that the energy management is properly set, such that the laptop's hard drive or operating system is not shutting down or going into real sleeping mode while executing the compilation process. 
#. If you already compiled parts of the project, read the hints for reversing these versions first.
#. Most of the required software packages can either be downloaded from the internet or be found in a sorted folder structure at the following Obelix server-address: ::
    
    \\Obelix\software\m\ITOM\Installationen
    
#. This document recommends to install certain software packages at given absolute file pathes. This is not necessary and you can easily replace the corresponding pathes by your individual structure. 


Installation procedure
----------------------

Step 1: Visual Studio 2010 Professional and Service Pack 1
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Install Visual Studio 2010 Professional.
* Install Service Pack 1 for Visual Studio 2010. The service pack is available under http://www.microsoft.com/download/en/details.aspx?id=23691

If you forget to install Service Pack 1, the compilation of itom on 64bit systems might fail. (see https://bugreports.qt.nokia.com//browse/QTBUG-11445) 

Step 2: Qt 4.8.0 for using together with Visual Studio 2010 Professional
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Since december 2011 you can directly download the Qt-version 4.8.0 from the Qt-website, which is precompiled for Visual Studio 2010. It is recommended to switch to this version.

* Download Qt libraries 4.8.0 for Windows (VS 2010) from the internet or get it from the Obelix server.
* Install it (e.g. under C:\\Qt). After installation you will have the folder C:\\Qt\\4.8.0.

Set the following environment variables in your windows system (if not available yet):

* entry QTDIR with value C:\\Qt\\4.8.0 (depending on the Qt's installation folder) 
* entry QMAKESPEC with value win32-msvc2010 (even if you are compiling for 64bit)
* add the following text to the Path variable: ;%QTDIR%\\bin (please add this string, do not replace the existing path-entry) 

These environment variables will firstly become available after a re-login of your user at the computer. Since you should add another environment variable in the section OpenCV, consider to install OpenCV first, which is independent from the Qt-installation.

If you are using Qt in a 32bit environment, go on with step 3.

Step 2.1: Compiling for 64bit
""""""""""""""""""""""""""""""

* Delete files beginning with sync from %QTDIR%\\bin directory (in order to avoid the requirement of Perl during compilation, which is not necessary in our case). 
* 64bit: Open Visual Studio Commandline x64 Win64 (2010) in your Start-Menu under Microsoft Visual Studio >> Visual Studio Tools. 
* change to Qt-Dir by cd %QTDIR%
* configure Qt-compilation by executing::
    
    configure -platform win32-msvc2010 -debug-and-release -opensource -no-qt3support -qt-sql-odbc -qt-sql-sqlite -qt-zlib -qt-libpng -webkit 
   
* choose open source version and accept the license information while configuration process. The configuration may take between 5 and 20 minutes. 
* now start the time-intense compilation process (1 to 5 hours) by typing::
    
    nmake
   
Step 3: Qt Add-In Visual Studio
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In order to install/run this add-in, be sure that the .NET framework with version 2.0 and SP1 is installed on your computer. It is not sufficient to have .NET with a version bigger than 2.0 installed! 

* Downlad Visual Studio Add-In qt-vs-addin-1.1.10.exe from the Qt-Webpage or get it from Obelix Server. 
* Install this Add-In (Visual Studio should be closed during installation)
* Now, you will find a new menu Qt in Visual Studio 

Step 4: Install and compile QScintilla2
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Just go to \\Obelix\\software\\m\\ITOM\\Installationen\\4. QScintilla2 and copy the folder **QScintilla2.6** to a directory on your hard drive (e.g. C:\\QScintilla2.6). The open your Visual Studio Command Line and change to the directory of **QScintilla** on your hard drive. Just execute the batch file ::
    
    qscintilla_install.bat
   
Step 5: Install OpenCV 2.3
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Download the OpenCV-Superpack (Version 2.3 or higher) from http://sourceforge.net/projects/opencvlibrary/files/opencv-win/2.3/ or Obelix server.
* The EXE-file is a self-extracting archive. Unpack it for example in folder C:\\OpenCV2.3
* The superpack is precompiled for VS2008, VS2010, MinGW in 32bit and 64bit. Therefore no further compilation has to be done.
* Depending on 32bit or 64bit and the development environment (here: VS2010), add the path to the appropriate bin-folder of OpenCV to the windows environment variable Path: 
    
    * VS2010, 32bit, ADD to the Path-variable: ;C:\\OpenCV2.3\\build\\x86\\vc10\\bin
    * VS2010, 64bit, ADD to the Path-variable: ;C:\\OpenCV2.3\\build\\x64\\vc10\\bin

* changes to the environment variable will become available after a re-login to the windows system!


Step 6: Install PointCloudLibrary binaries (1.5.1)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The PointCloud-Library is a sister-project of OpenCV and is able to work with large point clouds.

The binaries can be loaded from the website http://www.pointclouds.org/downloads/windows.html or from \\\\Obelix\\software\\m\\ITOM\\Installationen\\7. Point Cloud Library.

* Depending on 32bit or 64bit execute the AllInOne-Installer for Visual Studio 2010. The installation directory may for example be C:\\PCL1.5.1. Information: Please install the PCL base software including all 3rd-party packages, besides OpenNI. You don't have to install OpenNI, since this is only the binaries for the communication with commercial range sensors, like Kinect.
* If you want to debug into the point cloud library (not necessary, optional) unpack the appropriate zip-archive with the pdb-files into the bin-folder of the point cloud library. This is the folder where the dll's lie, too.

Step 7: Install Python 3.2
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Download the 32bit or 64bit-version of python from the website or Obelix server and install it.
* It is possible to have different versions of python installed in parallel on the same computer.

Step 8: Install NumPy
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Download the right versions


Step 9: Getting the |itom| project
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For getting the actual |itom| you have to download the project. The whole project is located in an SVN-Repository (versioning) and can be reached under: ::
    
    http://obelix.ito.uni-stuttgart.de/svn/itom/m12
    User: anonymous     
    Password: anonymous

Alternative access if you already have a SSH access to Obelix and you are in the SVN group: ::
    
    svn+ssh://ito[Nachname]@obelix/home/svn/itom/m12
    
Account, Username, Password
  
    The SVN-Archive can be downloaded (checkout) by all users (User: anonymous Password: anonymous). Everyone who wants to upload some files needs an activation by Heiko Bieger. The username is then the standard ito-Usernamer (ito[surname]) and the password can be chosen during the activation process.


Step 10: Adjustment of pathes in itom_properties.props
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Adjust the visual studio property file (itom_properties.props) with respect to the template itom_properties_template.props.


The project files for ITOM using Microsoft Visual Studio Professional 2010 use several times placeholder values (macros), which have the form $(Name). Most of these macros are given by the visual studio compiler.

In order to realize a simple compilation at different computers which different directory structures, we use a file with ending .props in order to add our custom macros for the project files. All our user defined macros have the form \*\*$(ITOM_Name).

If you newly downloaded ITOM you will find the file itom_properties_template.props in the folder trunk/iTOM. Please duplicate this file and rename it to itom_properties.props.

Edit itom_properties.props with a usual text editor or directly in Visual Studio 2010 and set the values of each item to your appropriate value. 

Example
""""""""""""""""""""""""""""""
The macro ITOM_QTDIR should have the value C:\\Qt\\4.8.0 and is found in 

.. code-block:: none
   
   <PropertyGroup Label="UserMacros">
      <ITOM_QTDIR>C:\Qt\4.8.0</ITOM_QTDIR>
      ...
   </PropertyGroup>

Macros
""""""""""""""""""""""""""""""

ITOM uses the following macros. All macros ending with 64 are values for 64-bit compilation only. If you don't compile in 64-bit you don't have to set their value to any appropriate string. 

* ITOM_QTDIR is the Qt-base directory, e.g. C:\\Qt\\4.8.0
* ITOM_PYTHONDIR is the python-base directory, e.g. C:\\Python32
* ITOM_OPENCVDIR is the directory to the build directory of OpenCV: C:\\OpenCV2.3\\build (siehe readme.txt von Open-CV)
* ITOM_QTDIR64 see ITOM_QTDIR for 64bit compilation
* ITOM_PYTHONDIR64 see ITOM_PYTHONDIR for 64bit compilation
* ITOM_OPENCV_VER is the version string of OpenCV, e.g. 230 for version 2.3.0
* ITOM_CUDA_PATH (optional) is the path to cuda, e.g. C:\\Program Files (x86)\\NVIDIA GPU Computing Toolkit\\CUDA\\v4.1
* ITOM_CUDA_PATH64 see ITOM_CUDA_PATH for 64bit compilation
* ITOM_PCLVER is the major and minor version of point cloud library (no patch), e.g. 1.5 or 1.6
* ITOM_PCLDIR is the path to the base directory of the point cloud library, e.g. C:\\PCL1.5.1
* ITOM_PCLDIR64 see ITOM_PCLDIR for 64bit compilation
* ITOM_BOOST_VER is the version string of boost, e.g. 1_47 for version 1.47 

These macros are listed, if everything is correct, in the visual studio project settings by clicking on the button Macros which is available in many sub-dialogs.


Direct-Adjustment of the property-file in Visual Studio Professional 2010
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

The property-file is created or added in the following way:

#. Visual-Studio: click View >> Property-Manager
#. For every project, the file itom_properties should be available
#. Double-Click on any of these files (changes may influence all projects) and click on tab User-defined macros
#. If the file is not available, you can add an existing property-file or create a new one by the toolbar above the property-manager in Visual Studio




