.. include:: ../include/global.inc

.. |mm| unicode:: U+00B5 m

Changelog Versions 5.x
######################

Version 5.0.0 (2025-MM-DD)
**************************

Itom-Project
------------

 * Added external git submodule

External
--------
 * This submodule comprises a CMake project to build a relevant external dependencies
 such as Qt, OpenCV Boost, VTK and PCL for all plattforms from within the itom-project.
 This is referred as ExternalProject and aims to overcome in particular the binary
 dependencies on Windows Platforms and should enable homogenous versions and code dependendcies
 across all plattforms.