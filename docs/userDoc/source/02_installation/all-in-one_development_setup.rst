.. include:: ../include/global.inc

.. _install-all-in-one-dev-setup:

All-In-One development setup
===============================

For users who want to get a development environment for itom, the main plugins and designer plugins there is an all-in-one development setup available.

Using this setup, you only need to unzip one or two archives to your harddrive, install **git** and **Python** that are included in this archive and
execute a setup script, written in Python. This script automatically downloads the current sources of |itom| and its plugins from the internet,
configures the 3rd party dependencies (also provided in this package) and automatically configures and generates CMake for the single repositories.
Using this setup tool, you can start developing |itom| or plugins within a short time.

The all-in-one development setup comes with the following features and 3rd party packages:

* Available for Visual Studio 2010 32bit and 64bit (Visual Studio 2010 and Service Pack 1 required)
* Git 1.9.4 (setup in package, needs to be installed)
* Python 3.4.2 (setup in package, needs to be installed)
* Numpy MKL 1.8.2 (setup in package, needs to be installed)
* Qt 5.3.2 (prebuild)
* OpenCV 2.4.10 (prebuild for VS2010)
* CMake 3.0.2 (prebuild)
* QScintilla 2.8 (prebuild inside of Qt 5.3.2)
* Doxygen

Optionally there is a 3rd party package that brings support for the PointCloudLibrary for itom. This secondary archive contains the following features:

* Boost 1.57.0 (prebuild)
* Eigen 3.0.5 (prebuild)
* Flann 1.7.1 (prebuild)
* QHull 2011.1 (prebuild)
* VTK 6.1.0 (prebuild with Qt 5.3.2 support)
* PCL 1.8.0 (prebuild with support of all libraries above)

Prerequisites for the development setup
****************************************

If you want to use the development setup the following prerequisites must be fulfilled:

* Windows 7 or higher (XP untested)
* Visual Studio 2010 + Service Pack 1 must be installed on the computer

Get and install the setup
**************************

The setup comes with one or two zip-archive files:

* **itom_development_VS2010_x86.zip** or **itom_development_VS2010_x64.zip** (required)
* **itom_development_VS2010_x86_PCLAddOn.zip** or **itom_development_VS2010_x64_PCLAddOn.zip** (optional, for compilation itom with PointCloudLibrary support, 3D visualization...)

Download the 32bit or 64bit version depending on your needs from https://sourceforge.net/projects/itom/files/all-in-one-build-setup/.

Then execute the following steps:

1. Download and unpack one or both archives into the same folder. You should then have the following folder structure:
    
    .. figure:: images/all-in-one-build/unzip-archives.png
        :scale: 100%
        :align: left

    * __install__
    * 3rdParty
    * 3rdPartyPCL (if you unpack the ..._PCLAddOn.zip as well
    
    The __install__ folder is only necessary during the installation and can be deleted afterwards. After the installation, this
    folder will also contain a folder **sources** and **build** with the source repositories of itom and their builds.
    
2. Go to the **__install__** folder
    
    .. figure:: images/all-in-one-build/things-to-install.png
        :scale: 100%
        :align: left

    If not yet available, install
    
    * Git 1.9.4
    * Python 3.4.2
    * Numpy 1.8.2
    
    on your computer. This is required for the further installation. Optionally you can install the **Qt-AddOn** for Visual Studio as well as
    **TortoiseGit** from the **optional** folder. This can also be done later.
    
3. Execute the script **setup.py**
    
    The next step is to execute the python script **setup.py** in the __install__ subfolder. Usually, this should work by double-clicking on the file. However
    you need to make sure that the file is executed with the Python 3.4.2 version that you recently installed. If this is not the case, make a right click on the file
    and select **Open with...** and choose **python.exe** from the Python installation path (e.g. C:/python34).
    
    This script leads you through the remaining installation process using a menu guided approach:
    
    .. figure:: images/all-in-one-build/setup_py_screenshot.png
        :scale: 100%
        :align: left

    Type any number (1-9) after the question **your input?** and press return to start the corresponding installation step. You can also execute the first six steps using the
    overall command number 7. Press 9 to quit the setup. You can continue with other steps by call **setup.py** again. Once you executed one step, an **(OK)** after the number
    shows you that the step already has been executed. An **(??)** means that no further information about a possible execution can be shown. The single step should normally be
    executed starting at 1 and ending with 8; however you can also omit single steps. The steps are explained in the following points.
    
    .. note::
        
        If you have already executed **setup.py** at least once, a file **setup_settings.txt** is created in the **__install__** folder. This contains some settings that you 
        made during the installation. Delete this file if you want to restart the installation without preset settings.
    
    1. Setup: clone git directories
    
        This command clones the **itom**, **plugins** and **designerPlugins** repositories from https://bitbucket.org/itom and creates the folders:
            
            * sources/itom
            * sources/plugins
            * sources/designerplugins
        
        If you have never executed this step before, the setup tries to guess the path to the application **git.exe** (that you installed before). You can accept the
        guessed path (type **y** and press return) or you can indicate the absolute path to this executable (e.g. **C:/Program Files (x86)/Git/bin/git.exe**). This setting
        is stored in the settings file **setup_settings.txt**.
        
    2. Patch Qt in 3rdParty folder
        
        The folder **3rdParty** contains a prebuild version of Qt (5.3.2, with OpenGL support). No further compilation needs to be done. However this Qt installation needs to be
        patched in order to correspond to your pathes. Use this setup step to execute the patch. If you are able to start executables like **assistant.exe** or **designer.exe** from the
        **qtbase/bin** folder of Qt, the patch seemed to work.
        
        .. note::
            
            **How is this Qt prebuild version created?**
            
            The Qt version in the prebuild setup is obtained by the source archive of Qt 5.3.2 from qt-projects.org. Using MSVC2010 32bit or 64bit, these sources have been configured using the same steps than indicated in :ref:`this link <compile-qt5-from-sources>`. The configured project is then compiled (using jom 1.0.14 for a multi-threaded compilation) and only the relevant files are then copied into the Qt5.3.2 folder of your prebuild itom setup.
            
        .. note::
            
            One drawback of this prebuild Qt installation is that you cannot directly debug into Qt methods (since the delivered ptb-files are not patched and point to invalid pathes). If
            you want to have this feature (usually not required for standard programming), you need to compile Qt by yourself into the same folder using the approach given in :ref:`this link <compile-qt5-from-sources>`.
    
    3. Configure and Generate CMake (itom only)
        
        At first, |itom| needs to be configured using CMake such that the folder **build/itom** is generated with an appropriate Visual Studio solution. CMake is directly called from **3rdParty/CMake3.0.2**. (If the configuration should fail, the CMake GUI is opened, you can reconfigure anything, generate the project by yourself and close the GUI. Then, the setup will continue.) Before you go on configuring the plugins and designerplugins, you need to build |itom| in Debug and Release first (in order to create the SDK). That is done in the following step.
        
    4. Compile **itom** in Debug and Release
        
        This step compiles itom from **build/itom** in a debug and release compilation. Then the executables **qitom.exe** and **qitomd.exe** are generated. If you try to start them, this may fail, since the pathes to the binaries of Qt, OpenCV and optionally the PointCloudLibrary are not included in the Windows path environment variable yet (see step 8)
    
    5. Configure and generate CMake of plugins and designerplugins
        
        Similar to step 3, the plugins and designerplugins are now configured and generated (folders **build/plugins** and **build/designerplugins**). This step will fail, if the itom-SDK could not be found in build/itom/SDK.
        
    6. Compile **plugins** and **designerplugins** in Debug and Release
        
        Same than step 4 but for plugins and designerPlugins.
        
    7. Bundle
        
        In order to execute steps 1-6 manually, you can execute all these steps in one row using command 7 (sometimes you need to accept intermediate steps by pressing return)
        
    8. Modify Windows path variable
        
        In order to find the binaries of Qt, OpenCV and optionally the PointCloudLibrary, it is necessary to prepend some pathes to the Windows path variable. If you choose option 8, a string is print to the command line and saved in the file **enver.txt**. Copy the string and **prepend** it to the PATH environment variable of Windows. Afterwards it is required to restart the computer or log-off and log-on again.
        
    Now you are done with the setup. If you want, you can delete the entire **__install__** folder.
    
    Try to execute **qitomd.exe** or **qitom.exe**.
