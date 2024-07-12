.. include:: ../include/global.inc
.. _build_cmake_vars_list:

List of common CMake Variables and Environment Variables used in the Itom project
=======================================================================================

This chapter is to hold lots of common CMAKE Variables and System Environment variables,
used throughout the itom and its plugin's projects, in the hope this will reduce the amount of
try-and error cycles to fill in the right patterns.

Environment Variables are set in the system environment and can be set in
via the command terminal.

**Linux/MaxOS**:
.. code-block:: bash

   export <VARIABLE_NAME>=<Value>

to change the environment variables permanently the user has to modify the shell profile
settings. The location of the corresponding file depends on the system.

.. code-block:: bash

   sudo echo "export <VARIABLE_NAME>=<Value>" >> ~/.profile


**Windows**:
.. code-block:: bash

   set <VARIABLE_NAME> <Value>

to change the environment variables permanently use:

.. code-block:: bash

   setx <VARIABLE_NAME> <Value>

In general it is recommended to use the system environment variables, because they can
be reused permanently and are used to set during the internal cmake configuration
to define the correct location of the cmake modules and configurations scripts, which
in turn define the set of Cmake Variables.

CMake Variables cam used to override configurations Cmake configurations
based on Environment variables. They can be set during the command line call
of the cmake configuration via the -D<Variable name> Flag (e.g. -D OpenCV_DIR),
or within the CMAKE-GUI.

This list is to be extended, but if you want to compile a certain plugin, best is to check with the
doc provided by it.

Environment Variables:
----------------------

.. |cmakelist_itom_sdk_root| replace::

   Path to the build ITOM SDK Path, as created after a successful |Itom| build.
   (e.g. <Itom_BuildFolder>/SDK). Should be set manually if ITOM is not build
   outside of the proposed build structure.

.. |cmakelist_qt_root| replace::

   Path to the compiler specific Qt folder (e.g. <QT_BuildFolder>/msvc####_##).

.. |cmakelist_python_root| replace::

   Path to the Python installation folder (e.g. C:/Python3).

.. |cmakelist_opencv_root| replace::

   Path to folder containing OpenCVConfig.cmake or
   OpenCVConfig-version.cmake  (e.g. <OpenCV_BuildFolder>/x64/vc##/lib).

.. |cmakelist_boost_root| replace::

   Path to the boost build folder (e.g. <BOOST_BuildFolder>).

.. |cmakelist_flann_root| replace::

   Path to the flann build folder (e.g. <FLANN_BuildFolder>).

.. |cmakelist_vtk_root| replace::

   The folder in the vtk module containing VTKConfig.cmake
   (e.g.  <VTK_BuildFolder>\lib\cmake\vtk-#.# )
   or VTKConfigVersion.cmake, highly depending on the vtk version you are using.
   On Linux Systems you need to separately install vtk-dev(even if it does not
   fit the version you want to use) for one or two headers that are missing otherwise...

.. |cmakelist_eigen_root| replace::

   Path to the eigen build folder (e.g. <EIGEN_BuildFolder>).

.. |cmakelist_pcl_root| replace::

   Path to the pcl build folder (e.g. <PCL_BuildFolder>\PCL1.12.0\cmake).

.. |cmakelist_libusb_root| replace::

   Path to the libusb build folder (e.g. <LIBUSB_BuildFolder>).

.. |cmakelist_fftw_root| replace::

   Path to the fftw build folder (e.g. <FFTW_BuildFolder>).

+-------------------+-------------------------------------------------------+
| Env. Variable     | Description                                           |
+===================+=======================================================+
| **ITOM_SDK_ROOT** | |cmakelist_itom_sdk_root|                             |
+-------------------+-------------------------------------------------------+
| **QT_ROOT**       | |cmakelist_qt_root|                                   |
+-------------------+-------------------------------------------------------+
| **PYTHON_ROOT**   | |cmakelist_python_root|                               |
+-------------------+-------------------------------------------------------+
| **OPENCV_ROOT**   | |cmakelist_opencv_root|                               |
+-------------------+-------------------------------------------------------+
| **BOOST_ROOT**    | |cmakelist_boost_root|                                |
+-------------------+-------------------------------------------------------+
| **FLANN_ROOT**    | |cmakelist_flann_root|                                |
+-------------------+-------------------------------------------------------+
| **VTK_ROOT**      | |cmakelist_vtk_root|                                  |
+-------------------+-------------------------------------------------------+
| **EIGEN_ROOT**    | |cmakelist_eigen_root|                                |
+-------------------+-------------------------------------------------------+
| **PCL_ROOT**      | |cmakelist_pcl_root|                                  |
+-------------------+-------------------------------------------------------+
| **LIBUSB_ROOT**   | |cmakelist_libusb_root|                               |
+-------------------+-------------------------------------------------------+
| **FFTW_ROOT**     | |cmakelist_fftw_root|                                 |
+-------------------+-------------------------------------------------------+

Cmake Variables:
----------------

.. |cmakelist_Itom_SDK_DIR| replace::

   Path to the build ITOM SDK Path, as created after a successful |Itom| build.
   (e.g. <Itom_BuildFolder>/SDK). Should be set manually if ITOM is not build
   outside of the proposed build structure.

.. |cmakelist_Qt_Prefix_DIR| replace::

   Path to the compiler specifice Qt build folder (e.g. <QT_BuildFolder>/msvc####_##).

.. |cmakelist_Python_ROOT_DIR| replace::

   Path to the Python installation folder (e.g. C:/Python3).

.. |cmakelist_build_qtversion| replace::

   Selected Qt Version. Currently Qt6 and Qt5 are supported. Qt6 i set to default.

.. |cmakelist_opencv_dir| replace::

   Path to folder containing OpenCVConfig.cmake or
   OpenCVConfig-version.cmake

.. |cmakelist_boost_dir| replace::

   Set this to the boost base folder you want to compile
   into itom. This folder contains folders "boost" and "lib64-msvc-xxx". If you
   set the right one here the boost libraries will get autopoulated...
   If not, this entry gets cleared again...

.. |cmakelist_flann_include_dirs| replace::

   Path to filder named "include"
   which lives in a directory side-by-side with "bin" and "lib" folders.

.. |cmakelist_vtk_dir| replace::

   The folder in the vtk module containing VTKConfig.cmake
   or VTKConfigVersion.cmake, highly depending on the vtk version you are using.
   On Linux Systems you need to separately install vtk-dev(even if it does not
   fit the version you want to use) for one or two headers that are missing otherwise...

.. |cmakelist_eigen_root| replace::

   Path to folder containing subfolders "build",
   "Eigen", "unsupported"

.. |cmakelist_pcl_dir| replace::

   Path to folder containing PCLConfig.cmake or
   PCLConfigVersion.cmake

.. |cmakelist_libusb_dir| replace::

   Use the libusb version from github. or at least 1.0.22
   using older versions requires lots more hassling on windows than downloading the git and compiling it.
   Believe me. Or try it out yourself. On the git version, this path points directly
   to the git(or downloaded and unpacked folder), named libusb or libusb-master
   or libusb-<branchname-you-checked-out> containing "libusb" subfolder and many more.

.. |cmakelist_fftw_dir| replace::

   Directory containing all the fftw libraries you
   want to use. You might need to compile some of them of you own,
   check with http://www.fftw.org/

.. |cmakelist_build_target| replace::

   Click this checkbox if you want to build itom for 64bit.
   Usually automatically detected depending on chosen compiler

.. |cmakelist_build_type| replace::

   Sets the output build type. only relevant for
   single configuration builds, as makefiles. not relevant for high-end IDEs like
   Visual Studio. Build Flags are set according to this selection.

.. |cmakelist_itom_sdk_dir| replace::

   Path to build/itom/SDK. If you allow you
   cmake projects use cached vars/load them from itom project, you can use the
   variables that are set in the itom project.

.. |cmakelist_vld_dir| replace::

   You can only set this value if compiling itom
   with Visual Studio in Debug mode. Point it to the directory similar to
   *D:\\itom\trunk\\Visual Leak Detector*. This folder must contain subfolders
   named bin, include and lib. Click **VISUALLEAKDETECTOR_ENABLED** in order to
   enable the memory leak detector in Visual Studio. Please make sure, that you
   add the correct subfolder of its bin directory to the windows environment variables
   or copy the content to the executable directory of itom (where qitom.exe is finally located).

+----------------------------+--------------------------------------------+
| Cmake Variable             | Description                                |
+============================+============================================+
| **ITOM_SDK_DIR**           | |cmakelist_Itom_SDK_DIR|                   |
+----------------------------+--------------------------------------------+
| **Qt_Prefix_DIR**          | |cmakelist_Qt_Prefix_DIR|                  |
+----------------------------+--------------------------------------------+
| **Python_ROOT_DIR**        | |cmakelist_Python_ROOT_DIR|                |
+----------------------------+--------------------------------------------+
| **BUILD_QTVERSION**        | |cmakelist_build_qtversion|                |
+----------------------------+--------------------------------------------+
| **OpenCV_DIR**             | |cmakelist_opencv_dir|                     |
+----------------------------+--------------------------------------------+
| **Boost_INCLUDE_DIR**      | |cmakelist_boost_dir|                      |
+----------------------------+--------------------------------------------+
| **FLANN_INCLUDE_DIRS**     | |cmakelist_flann_include_dirs|             |
+----------------------------+--------------------------------------------+
| **VTK_DIR**                | |cmakelist_vtk_dir|                        |
+----------------------------+--------------------------------------------+
| **Eigen_ROOT**              | |cmakelist_eigen_root|                      |
+----------------------------+--------------------------------------------+
| **PCL_DIR**                | |cmakelist_pcl_dir|                        |
+----------------------------+--------------------------------------------+
| **LibUSB_DIR**             | |cmakelist_libusb_dir|                     |
+----------------------------+--------------------------------------------+
| **FFTW_DIR**               | |cmakelist_fftw_dir|                       |
+----------------------------+--------------------------------------------+
| **BUILD_TARGET64**         | |cmakelist_build_target|                   |
+----------------------------+--------------------------------------------+
| **CMAKE_BUILD_TYPE**       | |cmakelist_build_type|                     |
+----------------------------+--------------------------------------------+
| **VISUALLEAKDETECTOR_DIR** | |cmakelist_vld_dir|                        |
+----------------------------+--------------------------------------------+
