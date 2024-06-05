.. include:: ../include/global.inc

.. _build-centos:

Build on CentOS
======================

This section describes how |itom| and its plugins are built on CentOS (tested with CentOS 7).
For general information about building |itom| under linux, see the :ref:`build instructions for linux <build-Debian>`.

In the following, all required steps are indicated to get the dependencies, get the sources and compile itom by the command line.
The folder structure is chosen to be equal than the one from the linux instructions, however itom will only be build in a release version.

Please execute the following commands in the command line to get the dependencies for |itom| (comments after the hash-tag should not be copied to the command line):

.. code-block:: bash

    sudo yum install epel-release
    sudo yum install cmake3 cmake3-gui
    sudo yum install git gcc gcc-c++
    sudo yum install python36 python36-devel python36-numpy
    sudo yum install opencv opencv-devel libv4l libv4l-devel
    sudo yum install qt5-qtbase-gui qt5-qtwebkit qt5-qtwebkit-devel
    sudo yum install qt5-qtsvg qt5-qtsvg-devel qt5-designer qt5-qttools-static qt5-qttools-devel
    sudo yum install libusbx-devel libusb-devel libudev-devel

These dependencies does not include support for point cloud libraries. To enable this support, you have to
get further packages. Please see :ref:`build on fedora <build-fedora>` for more hints.

Now, change to the base directory, where the sources and builds of itom and its plugins should be placed. The following commands are not executed
with super-user rights; prepend *sudo* if this is required. In comparison to building *itom* under Debian based Linux versions, the CMake
configuration process under CentOS needs some more *hints* about where to find some libraries etc. Therefore, it might be, that you have
to adjust some paths below. Currently, Qt5 is still built with webkit-support under CentOS, such that the WebEngine-support is not available.
Therefore, the built-in helpviewer of itom has to be disabled. For building itom **without** point cloud support use:

.. code-block:: bash

    git clone --recursive --remote git@github.com:itom-project/itomProject.git
    cd itomproject
    git submodule foreach --recursive git checkout master
    mkdir -p ./{build_debug,build_release}
    cd ./build_release
    cmake -G "Unix Makefiles" -DBUILD_WITH_PCL=OFF -PYTHON_LIBRARY=/usr/lib64/libpython3.6m.so -PYTHON_INCLUDE_DIR=/usr/include/python3.6m -Qt_Prefix_DIR=/usr/lib64 -BUILD_WITH_HELPVIEWER=OFF ../
    make -j4

Errors in cmake configuration process can be fixed using ccmake or cmake-gui
