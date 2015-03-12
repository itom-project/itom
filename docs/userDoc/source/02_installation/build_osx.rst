.. include:: ../include/global.inc

Build on Mac OS X
================

This section describes how |itom| and its plugins are built on a Apple Mac systems running OS X 10.7 or later (tested on OS X 10.10). The general approach is similar to the other documentation in the install chapter that are mainly focussed on Windows.

Xcode
-----

Xcode is the default IDE on OS X systems. It is also necessary to install Xcode to obtain the command line tools (including the compiler clang and the linker ld).

Since OSX Lion, the Xcode installation doesn't by default include the command line tools. Part of which a script called easy_install we will need to install some packages.

Once Xcode has been installed, run Xcode and then open the preferences. Select the Download section and the Components tab and install Command Line Tools from there.

Homebrew
--------

Most necessary packages can be obtained by the package manager of your solution. We will use Homebrew.

Open a terminal window and copy and paste 

.. code-block:: bash

    ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)" into the terminal

Homebrew can also be installed as described at http://brew.sh .

Necessary brew packages
-------------------

The following list describe packages that are required or recommended for building |itom| and how to install them using brew:

Required:

* **Qt** (it is also possible to build it from source http://qt-project.org/doc/qt-4.8/install-mac.html )
* Editor **QScintilla2**
* **git**
* **Cmake**
* **PointCloudLibrary** (if exists version 1.6 or better 1.7, else see http://pointclouds.org/downloads/linux.html or build it on your own, the point cloud library is optional!)
* **Doxygen** (doxygen, doxygen-gui)
* **glew**
* **fftw**

To install all in one rush run

.. code-block:: bash

	brew install git gcc python3 cmake qt pkg-config freetype libpng ffmpeg qscintilla2 glew pcl fftw doxygen

Python 3
--------

The default Python version on OS X is 2.x. Since |itom| is using Python 3.x you installed in the previous step but it is'nt recommended to replace version 2.x with 3.x.

To make Python 3.x and its tools available to the command line add it to the PATH with a command like

.. code-block:: bash

    sudo export PATH=/Library/Frameworks/Python.framework/Versions/3.4/bin:$PATH

Be sure to adapt the path as necessary. Especially be use to change the version number to the actually installed one. 

Now we need to install two packages using easy_install (again remeber to check your version number!):

.. code-block:: bash

    sudo easy_install-3.4 virtualenv
    sudo easy_install-3.4 pyparsing
    
Python dependencies
-------------------

It is now time to install the missing Python packages using the following bash command

.. code-block:: bash

    pip3 install ipython mpmath numpy scipy Pillow dateutil matplotlib

This will install the following packages:

* **iPython** 
* **NumPy**
* **SciPy**
* **Pillow**
* **dateutil**
* **Matplotlib**

OpenCV
------

OpenCV must be build by the user and can not be downloaded prebuild.

Go to http://opencv.org/downloads.html to download the current stable version of OpenCV.

Extract it to a location you want to keep it, i.e. ~/opencv/.

Open CMake, click Browse Source and navigate to your OpenCV folder.

Click Browse Build and navigate to your build Folder (you might have to create it before).

Click the configure button. You will be asked how you would like to generate the files. Choose Unix-Makefile from the Drop Down menu and Click OK. CMake will perform some tests and return a set of red boxes appear in the CMake Window.

Uncheck **BUILD_SHARED_LIBS**, uncheck **BUILD_TESTS**, add an **SDK path** to **CMAKE_OSX_SYSROOT**, it will look something like this **/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX10.10.sdk**. Add **x86_64** to **CMAKE_OSX_ARCHITECTURES**, this tells it to compile against the current system. Uncheck **WITH_1394**, uncheck **WITH_FFMPEG**.

Click generate.

In a terminal go to your OpenCV build directory, make and install the library:

.. code-block:: bash
    cd ~/opencv/build
    make
    sudo make install

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
        ...
    ./build_release # build folder for release makefiles of...
        ./itom      # ...core
        ...

To create all folders in your user directory in one step, call the following bash commands:

.. code-block:: bash
    mkdir ~/itom; cd ~/itom; mkdir ./sources; mkdir ./build_debug; mkdir build_release

Obtain the sources
--------------------

Clone at least the core repository of |itom| (bitbucket.org/itom/itom) as well as the open source plugin and designerPlugin repository into the corresponding subfolders of the **sources** folder. You can do this by using any git client or the command 

.. code-block:: bash
    cd sources
    git clone https://bitbucket.org/itom/itom.git
    git clone https://bitbucket.org/itom/plugins.git
    git clone https://bitbucket.org/itom/designerplugins.git

Configuration process
----------------------

Use **CMake** to create the necessary makefiles for debug and/or release:

1. Indicate the folder **sources/itom** as source folder
2. Indicate either the folder **build_debug/itom** or **build_release/itom** as build folder. If the build folder already contains configured makefiles, the last configuration will automatically loaded into the CMake gui.
3. Set the following variables:
    
    * **CMAKE_BUILD_TYPE** to either **debug** or **release**
    * **BUILD_TARGET64** to ON.
    * **BUILD_UNICODE** to ON if you want to build with unicode support (recommended)
    * **BUILD_WITH_PCL** to ON if you have the point cloud library available on your computer and want to compile |itom| with support for point clouds and polygon meshes.
    
4. Push the configure button
5. Usually, CMake should find most of the necessary third-party libraries, however you should check the following things:
    
    * OpenCV: OpenCV is located by the file **OpenCVConfig.cmake** in the directory **OpenCV_DIR**. Usually this is automatically detected in **~/opencv/build/OpenCVConfig.cmake**. If this is not the case, set **OpenCV_DIR** to the correct directory and press configure.
    * Python3: On OS X CMake always finds the Python version 2 as default version. This is wrong. Therefore set the following variables to the right pathes: **PYTHON_EXECUTABLE** to **/Library/Frameworks/Python.framework/Versions/3.4/bin**, **PYTHON_INCLUDE_DIR** to **/Library/Frameworks/Python.framework/Versions/3.4/include**, **PYTHON_LIBRARY** to **/Library/Frameworks/Python.framework/Versions/3.4/lib/libpython3.4.dylib**. The suffix 3.4 might also be different. It is also supported to use any other version of Python 3.
6. Push the configure button again and then generate.
7. Now you can build |itom| by the **make** command:
    
    * **make**: Open a command line and switch to the **build_debug/itom** or **build_release/itom** directory. Simply call **make** such that the file **qitom** or **qitomd** (debug) is built. Start this application by calling **./qitom** or **./qitomd** in order to run |itom|.
    
Build plugins
---------------

Build the plugins, designerPlugins... in the same way than |itom| but make sure that you compiled |itom| at least once before you start configuring and compiling any plugin. In CMake, you need to indicate the same variables than above, but you also need to set the variable **ITOM_DIR_SDK** to the **sdk** folder in **build_debug/itom/sdk** or **build_release/itom/sdk** depending whether you want to compile a debug or release version (please don't forget to set **CMAKE_BUILD_TYPE**. 

If you don't want to have some of the plugins, simply uncheck them in CMake under the group **Plugin**.

The plugins and designerPlugins will finally be compiled and then copy their resulting library files into the **designer** and **plugins** subfolder of |itom|. Restart |itom| and you the plugin will be loaded.

PyPort bug
----------
If you get build errors that trace back to an error like

.. code-block:: bash
    ... /__locale:436:15: error: C++ requires a type specifier for all declarations
        char_type toupper(char_type __c) const
                        ^~~~~~~~~~~~~~~~~~~~~~
    /opt/local/Library/Frameworks/Python.framework/Versions/2.7/Headers/pyport.h:731:29: note: expanded from macro 'toupper'

You are using a deprecated version of PyPort.

Locate **pyport.h**, it might be located in **/Library/Frameworks/Python.framework/Versions/3.4/include/python3.4m/pyport.h**. Open it and replace

.. code-block:: c

    #ifdef _PY_PORT_CTYPE_UTF8_ISSUE
        #include <ctype.h>
        #include <wctype.h>
        #undef isalnum
        #define isalnum(c) iswalnum(btowc(c))
        #undef isalpha
        #define isalpha(c) iswalpha(btowc(c))
        #undef islower
        #define islower(c) iswlower(btowc(c))
        #undef isspace
        #define isspace(c) iswspace(btowc(c))
        #undef isupper
        #define isupper(c) iswupper(btowc(c))
        #undef tolower
        #define tolower(c) towlower(btowc(c))
        #undef toupper
        #define toupper(c) towupper(btowc(c))
    #endif

with

.. code-block:: c

    #ifdef _PY_PORT_CTYPE_UTF8_ISSUE
        #ifndef __cplusplus
            /* The workaround below is unsafe in C++ because
             * the <locale> defines these symbols as real functions,
             * with a slightly different signature.
             * See python issue #10910
             */
            #include <ctype.h>
            #include <wctype.h>
            #undef isalnum
            #define isalnum(c) iswalnum(btowc(c))
            #undef isalpha
            #define isalpha(c) iswalpha(btowc(c))
            #undef islower
            #define islower(c) iswlower(btowc(c))
            #undef isspace
            #define isspace(c) iswspace(btowc(c))
            #undef isupper
            #define isupper(c) iswupper(btowc(c))
            #undef tolower
            #define tolower(c) towlower(btowc(c))
            #undef toupper
            #define toupper(c) towupper(btowc(c))
        #endif
    #endif

See also http://bugs.python.org/review/10910/diff/8559/Include/pyport.h

