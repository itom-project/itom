:orphan:

.. include:: ../../include/global.inc

.. _build_dependencies_opencv:

Compile OpenCV
===============

OpenCV must be build from the source files. You should create following folder 
structure: **source, build_x64, install_x64**. Than open CMake and set the source 
and build path. Configure the CMake file with following options:

* CMAKE_INSTALL_PREFIX: absolute path to **install_x64**.

* BUILD options:

.. figure:: ../images/all-in-one-create/CMake_BUILD_OPENCV.png
   :scale: 100%
   :align: center
       
WITH options:

.. figure:: ../images/all-in-one-create/CMake_WITH_OPENCV.png
   :scale: 100%
   :align: center

Start the compilation of the **INSTALL** build solution.  
   
optional install CUDA Toolkit (e.g. 7.0, supported by OptiX as well). Is need 
for Macrosim, which runs with |itom|. Delete the executable (\*.exe) from the 
install_x64/x64/vc14/bin folder. They are not needed for the compilation of |itom|. 

.. note::

    Check the entry **BUILD_opencv_world** to combine all modules in one dll-file. 
