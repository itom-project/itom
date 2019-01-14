.. include:: ../include/global.inc

.. _build-linux:

Build on linux
================

This section describes how |itom| and its plugins are built on a linux system (tested on Fedora 25, Ubuntu 12.04 (32bit), Kubuntu (Debian, KDE, 64bit) and Lubuntu). 
The general approach is similar to the other documentation in the install chapter that are mainly focussed on Windows. Links to details about commands
necessary to obtain required packages, get the itom sources and build it, are given in the last section of this page.

Necessary packages
-------------------

Most necessary packages can be obtained by the package manager of your solution (Synaptic Package Manager, command *sudo apt-get*...).
The following list describe packages that are required or recommended for building |itom|:

Required:

* **Qt4** or **Qt5** (libqtcore4, libqt4-dev, libqt4-...)
* **OpenCV** (libopencv-core2.3 or libopencv-core2.4, libopencv-core-dev, libopencv-imgproc, libopencv-highgui...)
* **Python3** (python3, python3-dev, python3-dbg)
* **Numpy** (python3-numpy, python3-numpy-dbg)
* **git** (git)
* **Cmake** (cmake, cmake-gui)

Recommended (optional):

* The IDE **QtCreator** (qtcreator)
* **PointCloudLibrary** (if exists version 1.6 or better 1.7, else see http://pointclouds.org/downloads/linux.html or build it on your own, the point cloud library is optional!)
* **Scipy** (python3-scipy, python3-scipy-dbg), **Sphinx** (python3-sphinx), **Matplotlib**
* **Doxygen** (doxygen, doxygen-gui)
* Any git client (e.g. SmartGit (requires java) or git-cola)
* glew (libglew1.6-dev or something similar, required by some plugins)
* fftw (libfftw3-dev or something similar, required by some plugins)

for Ubuntu 12.04 based distributions the following command should install all necessary packages and its dependencies:

.. code-block:: bash

    sudo apt-get install git libqt4-dev libopencv-dev libopencv-highgui-dev python3-dev python3-dbg qtcreator python3-scipy-dbg python3-sphinx doxygen-gui libglew-dev cmake-gui qt4-dev-tools libcv-dev libhighgui-dev
    
for Ubuntu 16.04 based distributions the following command should install all necessary packages and its dependencies:

.. code-block:: bash

    sudo apt install cmake cmake-gui git
    sudo apt install python3 python3-dev python3-numpy python3-pip python3-apt-dbg
    sudo apt install libopencv-dev
    sudo apt install libqt5webkit5 libqt5webkit5-dev libqt5widgets5 libqt5xml5 libqt5svg5 libqt5svg5-dev libqt5gui5 libqt5designer5 libqt5concurrent5
    sudo apt install qttools5-dev-tools qttools5-dev
    sudo apt-get install libv4l-dev #this is optional to get the video for linux drivers
    sudo apt-get install xsdcxx libxerces-c3.1 libxerces-c-dev #this is optional to being able to compile the x3p plugin

for Ubuntu 18.04 based distributtions the following command should install all necessary packages and its dependancies:

.. code-block:: bash

    sudo apt install cmake cmake-gui git
    sudo apt install python3 python3-dev python3-numpy python3-pip python3-apt-dbg
    sudo apt install libopencv-dev
    sudo apt install libqt5webkit5 libqt5webkit5-dev libqt5widgets5 libqt5xml5 libqt5svg5 libqt5svg5-dev libqt5gui5 libqt5designer5 libqt5concurrent5
    sudo apt install qtwebengine5-dev libqt5webenginewidgets5 libqt5webengine-data
    sudo apt install qttools5-dev-tools qttools5-dev
    sudo apt update && sudo apt-get install build-essential
    sudo apt install fftw-dev libfftw3-single3 libfftw3-double3 libfftw3-long3 libfftw3-dev #this is optional for being able to compile fftw plugins
    sudo apt install libv4l-dev #this is optional to get the video for linux drivers
    sudo apt install xsdcxx libxerces-c3.2 libxerces-c-dev #this is optional for being able to compile the x3p plugin

If you want to compile with support from Point Cloud Library, also get the following packages:

.. code-block:: bash

    sudo apt-get install libpcl-dev libproj-dev
    

    

Recommended folder structure
-----------------------------

Similar to Windows, the following folder structure is recommended:

.. code-block:: python
    
    ./sources
        ./itom    # cloned repository of core
        ./plugins # cloned sources of plugins
        ./designerPlugins # cloned sources of designerPlugins
        ...
    ./build_debug # build folder for debug makefiles of...
        ./itom    # ...core
        ./plugins # ...plugins
        ./designerPlugins # ... designer plugins
        ...
    ./build_release # build folder for release makefiles of...
        ./itom      # ...core
        ./plugins # ...plugins
        ./designerPlugins # ... designer plugins
        ...

Under linux, the debug and release versions are separated in two different build folders. If you are using QtCreator as IDE you can however create two different configurations, one mapping to the debug build folder, the other mapping ot the release build folder. Both use the same source folder.

Obtain the sources
--------------------

Clone at least the core repository of |itom| (bitbucket.org/itom/itom) as well as the open source plugin and designerPlugin repository into the corresponding subfolders of the **sources** folder. You can do this by using any git client or the command **git clone https://bitbucket.org/itom/itom**).

Configuration process
----------------------

Use **CMake** to create the necessary makefiles for debug and/or release:

1. Indicate the folder **sources/itom** as source folder
2. Indicate either the folder **build_debug/itom** or **build_release/itom** as build folder. If the build folder already contains configured makefiles, the last configuration will automatically loaded into the CMake gui.
3. Set the following variables:
    
    * **CMAKE_BUILD_TYPE** to either **debug** or **release**
    * **BUILD_TARGET64** to ON if you want to build a 64bit version.
    * **BUILD_UNICODE** to ON if you want to build with unicode support (recommended)
    * **BUILD_WITH_PCL** to ON if you have the point cloud library available on your computer and want to compile |itom| with support for point clouds and polygon meshes.
    
4. Push the configure button
5. Usually, CMake should find most of the necessary third-party libraries, however you should check the following things:
    
    * OpenCV: OpenCV is located by the file **OpenCVConfig.cmake** in the directory **OpenCV_DIR**. Usually this is automatically detected in **usr/share/OpenCV**. If this is not the case, set **OpenCV_DIR** to the correct directory and press configure.
    * Python3: On some linux distributions, CMake always finds the Python version 2 as default version. This is wrong. Therefore set the following variables to the right pathes: PYTHON_EXECUTABLE to /usr/bin/python3.2, PYTHON_INCLUDE_DIR to /usr/include/python3.2, PYTHON_LIBRARY to /usr/lib/libpython3.2mu.so.1.0 . The suffix 1.0 might also be different. It is also supported to use any other version of Python 3.
6. Push the configure button again and then generate.
7. Now you can build |itom| by the **make** command or using **QtDesigner**:
    
    * **make**: Open a command line and switch to the **build_debug/itom** directory. Simply call **make** such that the file **qitom** or **qitomd** (debug) is built. Start this application in order to run |itom|.
    * **QtCreator**: Open QtCreator and open a new project. Indicate the file **CMakeList.txt** of **sources/itom** as project file. Now QtCreator asks where to build the binaries. Indicate the existing and pre-configured directory **build_debug/itom** or **build_release/itom**. The existing CMake cache files will be read and you can simply run CMake from QtCreator that should not fail. If so, re-open CMake and fix it. In the project settings of QtCreator you can finally clone the current configuration and indicate the second build-folder for the release version.
    
Build plugins
---------------

Build the plugins, designerPlugins... in the same way than |itom| but make sure that you compiled |itom| at least once before you start configuring and compiling any plugin. In CMake, you need to indicate the same variables than above, but you also need to set the variable **ITOM_SDK_DIR** to the **sdk** folder in **build_debug/itom/sdk** or **build_release/itom/sdk** depending whether you want to compile a debug or release version (please don't forget to set **CMAKE_BUILD_TYPE**). 

If you don't want to have some of the plugins, simply uncheck them in CMake under the group **Plugin**.

The plugins and designerPlugins will finally be compiled and then copy their resulting library files into the **designer** and **plugins** subfolder of |itom|. Restart |itom| and you the plugin will be loaded.
If **itom** is build with Point Cloud Library and you want to build the **Vtk3dVisualizer** with **vtk 6.2** you may come up with a linker exception: **cannot find -lvtkproj4**
This can be fixed by generating a sybolic link to any vtk .so file as followed:

.. code-block:: bash

    ln -s /usr/lib/x86_64-linux-gnu/libvtkCommonCore-6.2.so /usr/lib/libvtkproj4.so



Bash-based build
-----------------

You can also get all packages, download the sources and build itom including all plugins via the bash or command line. Instructions for Debian or Ubuntu systems having Qt5 packages in the respository
can be read at the :ref:`Raspberry Pi <build-raspi>` build instruction page. For Fedora see :ref:`Fedora build page <build-fedora>`.
    

