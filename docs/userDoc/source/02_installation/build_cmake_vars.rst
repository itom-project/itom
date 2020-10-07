.. include:: ../include/global.inc
.. _build_cmake_vars_list:

List of common **cmake** Variables used in the |itom| project
===============================================================

This chapter is to hold lots of common CMAKE Variables, used throughout 
the itom and its plugin's projects, in the hope this will reduce the amount of 
try-and error cycles to fill in the right patterns. This list is to be 
extended, but if you want to compile a certain plugin, best is to check with the 
doc provided by it.

.. |cmakelist_boost_dir| replace:: set this to the boost base folder you want to compile 
   into itom. This folder contains folders "boost" and "lib64-msvc-xxx". If you 
   set the right one here the boost libraries will get autopoulated... 
   If not, this entry gets cleared again...

.. |cmakelist_boost_includedir| replace:: from FindBoost.cmake:
   "preferred include directory e.g. .../include"
   create this entry if you have multiple boost versions installed and point it to the 
   include directory(the folder containing **headers**) of the right version.
   Note the all-caps spelling.

.. |cmakelist_boost_librarydir| replace:: from FindBoost.cmake:
   "Preferred library directory e.g. .../lib"
   create this entry and point it to the folder containing the 
   **compiled** boost libraries(\*.dll, \*.lib, \*.a, \*.so ...)
   Note the all-caps spelling.

.. |cmakelist_build_type| replace:: sets the output build type. only relevant for 
   single configuration builds, as makefiles. not relevant for high-end IDEs like 
   Visual Studio. Build Flags are set according to this selection.

.. |cmakelist_vtk_dir| replace:: the folder in the vtk module containing VTKConfig.cmake
   or VTKConfigVersion.cmake, highly depending on the vtk version you are using.
   On Linux Systems you need to separately install vtk-dev(even if it does not 
   fit the version you want to use) for one or two headers that are missing otherwise...
    
.. |cmakelist_itom_sdk_dir| replace:: path to build/itom/SDK. If you allow you 
   cmake projects use cached vars/load them from itom project, you can use the 
   variables that are set in the itom project.
    
.. |cmakelist_opencv_dir| replace:: path to folder containing OpenCVConfig.cmake or
   OpenCVConfig-version.cmake
    
.. |cmakelist_pcl_dir| replace:: path to folder containing PCLConfig.cmake or 
   PCLConfigVersion.cmake
    
.. |cmakelist_libusb_dir| replace:: use the libusb version from github. or at least 1.0.22
   using older versions requires lots more hassling on windows than downloading the git and compiling it.
   Believe me. Or try it out yourself. On the git version, this path points directly
   to the git(or downloaded and unpacked folder), named libusb or libusb-master
   or libusb-<branchname-you-checked-out> containing "libusb" subfolder and many more.

.. |cmakelist_fftw_dir| replace:: directory containing all the fftw libraries you 
   want to use. You might need to compile some of them of you own, 
   check with http://www.fftw.org/

.. |cmakelist_eigen_dir| replace:: path to folder containing subfolders "build", 
   "Eigen", "unsupported"
    
.. |cmakelist_qt5_dir| replace:: path containing "Qt5Config.cmake", 
   "Qt5ConfigVersion.cmake" and "Qt5ModuleLocation.cmake", usually in qt5 build folder, 
   msvcxxxx_xx/lib/cmake
    
.. |cmakelist_flann_include_dirs| replace:: path to filder named "include" 
   which lives in a directory side-by-side with "bin" and "lib" folders.
   
.. |cmakelist_vld_dir| replace:: you can only set this value if compiling itom
   with Visual Studio in Debug mode. Point it to the directory similar to
   *D:\\itom\trunk\\Visual Leak Detector*. This folder must contain subfolders
   named bin, include and lib. Click **VISUALLEAKDETECTOR_ENABLED** in order to
   enable the memory leak detector in Visual Studio. Please make sure, that you
   add the correct subfolder of its bin directory to the windows environment variables
   or copy the content to the executable directory of itom (where qitom.exe is finally located).

Liste
---------

    +----------------------------+--------------------------------------------------------------------------------------------------------------------------------------+
    | Variable                   | Description                                                                                                                          |
    +============================+======================================================================================================================================+
    | **OpenCV_DIR**             | |cmakelist_opencv_dir|                                                                                                               |
    +----------------------------+--------------------------------------------------------------------------------------------------------------------------------------+
    | **PCL_DIR**                | |cmakelist_pcl_dir|                                                                                                                  |
    +----------------------------+--------------------------------------------------------------------------------------------------------------------------------------+
    | **BUILD_TARGET64**         | click this checkbox if you want to build itom for 64bit. Usually automatically detected depending on chosen compiler                 |
    +----------------------------+--------------------------------------------------------------------------------------------------------------------------------------+
    | **CMAKE_BUILD_TYPE**       | |cmakelist_build_type|                                                                                                               |
    +----------------------------+--------------------------------------------------------------------------------------------------------------------------------------+
    | **VISUALLEAKDETECTOR_DIR** | |cmakelist_vld_dir|                                                                                                                  |
    +----------------------------+--------------------------------------------------------------------------------------------------------------------------------------+
    | **Boost_INCLUDE_DIR**      | |cmakelist_boost_dir|                                                                                                                |
    +----------------------------+--------------------------------------------------------------------------------------------------------------------------------------+
    | **BOOST_INCLUDEDIR**       | |cmakelist_boost_includedir|                                                                                                         |
    +----------------------------+--------------------------------------------------------------------------------------------------------------------------------------+
    | **BOOST_LIBRARYDIR**       | |cmakelist_boost_librarydir|                                                                                                         |
    +----------------------------+--------------------------------------------------------------------------------------------------------------------------------------+
    | **LibUSB_DIR**             | |cmakelist_libusb_dir|                                                                                                               |
    +----------------------------+--------------------------------------------------------------------------------------------------------------------------------------+
    | **Eigen_DIR**              | |cmakelist_eigen_dir|                                                                                                                |
    +----------------------------+--------------------------------------------------------------------------------------------------------------------------------------+
    | **Qt5_DIR**                | |cmakelist_qt5_dir|                                                                                                                  |
    +----------------------------+--------------------------------------------------------------------------------------------------------------------------------------+
    | **FFTW_DIR**               | |cmakelist_fftw_dir|                                                                                                                 |
    +----------------------------+--------------------------------------------------------------------------------------------------------------------------------------+
    | **FLANN_INCLUDE_DIRS**     | |cmakelist_flann_include_dirs|                                                                                                       |
    +----------------------------+--------------------------------------------------------------------------------------------------------------------------------------+
    | **VTK_DIR**                | |cmakelist_vtk_dir|                                                                                                                  |
    +----------------------------+--------------------------------------------------------------------------------------------------------------------------------------+

