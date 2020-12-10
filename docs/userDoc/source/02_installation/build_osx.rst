.. include:: ../include/global.inc

Build on Mac OS X
=================

This section describes how |itom| and its plugins are built on a Apple Mac systems running OS X 10.7 or later (tested on OS X 10.9 and 10.10). The general approach is similar to the other documentation in the install chapter that are mainly focussed on Windows.

Xcode
-----

Xcode is the default IDE on OS X systems. It is also necessary to install Xcode to obtain the command line tools (including the compiler clang and the linker ld).

Since OSX Lion, the Xcode installation doesn't by default include the command line tools. Part of which a script called easy_install we will need to install some packages.

Once Xcode has been installed, run Xcode and then open the preferences. Select the Download section and the Components tab and install Command Line Tools from there.

CMake
-----

Go to **http://www.cmake.org** to download and install the current stable releasse of CMake. It is recommended to download a release that looks like **cmake-x.y.z-Darwin-x86_64.dmg**

Homebrew
--------

Most necessary packages can be obtained by the package manager of your solution. We will use Homebrew.

Open a terminal window and copy and paste the following line into a terminal

.. code-block:: bash

    ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"

Run the following command once before installing anything

.. code-block:: bash

    brew doctor

Homebrew can also be installed as described at http://brew.sh .

Dependencies
------------

The following list describe packages that are required or recommended for building |itom| and how to install them using brew.

Required packages:

* **Qt**
* **git**
* **Cmake**

Recommended packages:

* **PointCloudLibrary**
* **Doxygen**
* **glew**
* **fftw**

Required Python packages:

* **NumPy**

Recommended Pyton packages:

* **SciPy**
* **Pillow**
* **Matplotlib**
* **sphinx**
* **frosted**

.. note::
    
    You will find the script *osx_install_dependencies.sh* in the source directory. This script allows to install all dependencies in one rush. To use it run the command *sh osx_install_dependencies.sh* in a Terminal.

To install all in one rush run the follwing list of commands. This might take a whole lot of time (we might talk about hours).

.. code-block:: bash

    brew update
    brew tap homebrew/science
    brew tap homebrew/python
    brew install git gcc python3 pkg-config
    brew install qt --with-developer --with-docs
    brew install pyqt --with-python3
    brew install doxygen --with-doxywizard --with-graphviz
    brew install ffmpeg glew fftw
    brew install numpy --with-python3
    brew link numpy --overwrite
    brew install pillow --with-python3
    brew link pillow --overwrite
    brew install matplotlib --with-python3 --with-pyqt
    brew link matplotlib --overwrite
    brew install matplotlib-basemap --with-python3
    brew link matplotlib-basemap --overwrite
    brew install scipy --with-python3
    brew link scipy --overwrite
    brew install opencv pcl caskroom/cask/brew-cask
    brew cask install qt-creator
    brew linkapps

The above commands in one line:

.. code-block:: bash

    brew update; brew tap homebrew/science; brew tap homebrew/python; brew install git gcc python3 pkg-config; brew install qt --with-developer --with-docs; brew install doxygen --with-doxywizard --with-graphviz; brew install pyqt --with-python3; brew install qscintilla2 --with-python3; brew install ffmpeg glew fftw; brew install numpy --with-python3; brew link numpy --overwrite; brew install pillow --with-python3; brew link pillow --overwrite; brew install matplotlib --with-python3 --with-pyqt; brew link matplotlib --overwrite; brew install matplotlib-basemap --with-python3; brew link matplotlib-basemap --overwrite; brew install scipy --with-python3; brew link scipy --overwrite; brew install opencv pcl caskroom/cask/brew-cask; brew cask install qt-creator; brew linkapps

If you would like to compile |itom| with Qt 5 replace *brew install qt --with-developer --with-docs* with *brew install qt5 --with-developer --with-docs* and *brew install pyqt --with-python3* with *brew install pyqt5 --with-python* .

Python 3
--------

The default Python version on OS X is 2.x. Since |itom| is using Python 3.x you installed in the previous step but it is'nt recommended to replace version 2.x with 3.x. We will set an alias for python3, so when entered python in a terminal session, python3 will be called.

To edit you aliases execute the following command.

.. code-block:: bash

    printf "alias python='python3'\n" >> ~/.bash_profile

The same thing must be done for *pip* and *easy_install*. Be adviced to check the installed version number of python and change it when necessary. The command python --version will give you the installed version number.

.. code-block:: bash

    printf "alias easy_install='/usr/local/Cellar/python3/3.4.3/bin/easy_install-3.4'\n" >> ~/.bash_profile
    printf "alias pip='/usr/local/Cellar/python3/3.4.3/bin/pip3.4'\n" >> ~/.bash_profile
    . ~/.bash_profile

Now we need to install two packages using easy_install (again remeber to check your version number!):

.. code-block:: bash

    sudo easy_install virtualenv pyparsing frosted

We need some more python packages. Just run the following command:

.. code-block:: bash

    pip install ipython mpmath sphinx 

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
    
    mkdir ~/itom; cd ~/itom; mkdir ./sources; mkdir ./build_debug; mkdir ./build_debug/itom; mkdir ./build_debug/plugins; mkdir ./build_debug/designerPlugins; mkdir ./build_release; mkdir ./build_release/itom; mkdir ./build_release/plugins; mkdir ./build_release/designerPlugins

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

1.  Indicate the folder **sources/itom** as source folder
2.  Indicate either the folder **build_debug/itom** or **build_release/itom** as build folder. If the build folder already contains configured makefiles, the last configuration will automatically loaded into the CMake gui.
3. Check Advanced to *see* all available options.
4.  Set the following variables:

    * **Itom Parameter**:
        * **BUILD_TARGET64** to ON.
        * **BUILD_WITH_PCL** to ON if you have the point cloud library available on your computer and want to compile |itom| with support for point clouds and polygon meshes.
    
    * **System Parameter**:
        * **CMAKE_BUILD_TYPE** to either **debug** or **release**
        * **CMAKE_OSX_ARCHITECTURES**: *x86_64*
        * **CMAKE_OSX_SYSROOT**: */Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX10.9.sdk*
        
        The suffix SDK version might also be different as well as the path.

5.  Push the configure button
6.  Usually, CMake should find most of the necessary third-party libraries, however you should check the following things:
  
    * **OpenMP**: OpenMP is not available when using the default compiler clang. If your setup includes OpenMP set *-fopenmp* to *OpenMP_CXX_FLAGS* and *OpenMP_C_FLAGS*.

    * **Python3**: On OS X CMake always finds the Python version 2 as default version. This is wrong. Therefore set the following variables to the right pathes: 

        * **PYTHON_EXECUTABLE**: /usr/local/bin/python3.4
        * **PYTHON_INCLUDE_DIR**: /usr/local/Cellar/python3/3.4.3/Frameworks/Python.framework/Versions/3.4/include/python3.4m
        * **PYTHON_LIBRARY**: /usr/local/Cellar/python3/3.4.3/Frameworks/Python.framework/Versions/3.4/lib/libpython3.4.dylib
        * **PYTHON_LIBRARY_DEBUG**: /usr/local/Cellar/python3/3.4.3/Frameworks/Python.PYTHON_LIBRARY_RELEASE**: framework/Versions/3.4/lib/libpython3.4.dylib

    The suffix 3.4 might also be different. It is also supported to use any other version of Python 3.

    * **OpenCV**:
        * **OpenCV_BIN_DIR**: /../bin
        * **OpenCV_CONFIG_PATH**: /usr/local/share/OpenCV
        * **OpenCV_DIR**: /usr/local/share/OpenCV

    * **PointCloudLibrary** (optonal):
        * **PCL_COMMON_INCLUDE_DIR**: /usr/local/include/pcl-1.7
        * **PCL_COMMON_LIBRARY**: /usr/local/lib/libpcl_common.dylib
        * **PCL_COMMON_LIBRARY_DEBUG**: /usr/local/lib/libpcl_common.dylib
        * **PCL_COMMON_DIR**: /usr/local/share/pcl-1.7

    The version 1.7 might also be different. It is also supported to use any other version of the PointCloudLibrary.

7.  Push the generate button.
8.  Now you can build |itom| by the **make** or using **Xcode** command:
    
    * **make**: Open a command line and switch to the **build_debug/itom** or **build_release/itom** directory. Simply call **make** such that the file **qitom** or **qitomd** (debug) is built. Start this application by calling **./qitom** or **./qitomd** in order to run |itom|.

    * **Xcode**: If you plan to change something in the source code it is recommended to use Xcode. Just open the **itom.xcodeproj** with Xcode and compile the project.

Build plugins
---------------

Build the plugins, designerPlugins... in the same way than |itom| but make sure that you compiled |itom| at least once before you start configuring and compiling any plugin. In CMake, you need to indicate the same variables 
than above, but you also need to set the variable **ITOM_SDK_DIR** to the **sdk** folder in **build_debug/itom/sdk** or **build_release/itom/sdk** depending whether you want to compile a debug or release version (please don't forget to set **CMAKE_BUILD_TYPE**. 

If you don't want to have some of the plugins, simply uncheck them in CMake under the group **Plugin**.

The plugins and designerPlugins will finally be compiled and then copy their resulting library files into the **designer** and **plugins** subfolder of |itom|. Restart |itom| and you the plugin will be loaded.

Known Problems
--------------

PyPort bug
^^^^^^^^^^

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

python3 not available in Terminal
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To make Python 3.x and its tools available to the command line add it to the PATH with a command like

.. code-block:: bash

    sudo export PATH=/Library/Frameworks/Python.framework/Versions/3.4/bin:$PATH

Be sure to adapt the path as necessary. Especially be use to change the version number to the actually installed one.

pip install fails with *TypeError: unorderable types: str() < NoneType()*
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Open /usr/local/Cellar/python3/3.4.3/libexec/setuptools/__init__.py and change 

.. code-block:: python

    self.py_version,
    self.platform,

for 

.. code-block:: python

    self.py_version or '',
    self.platform or '',

Do the same for /usr/local/lib/python3.4/site-packages/setuptools-12.2-py3.4.egg/pkg_resources/__init__.py

Source: https://bitbucket.org/minrk/setuptools/commits/e7d8f68ea7c603638562cf8278daa5d16e699f4e