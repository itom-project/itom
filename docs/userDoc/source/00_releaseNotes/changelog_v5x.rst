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
  such as Qt, OpenCV Boost, VTK and PCL for all platforms from within the itom-project.
  This is referred as ExternalProject and aims to overcome in particular the binary
  dependencies on Windows Platforms and should enable homogeneous versions and code dependencies
  across all platforms.

itom
----

* Script editor windows can be docked and undocked by drag and drop. Tabs can be dragged to other script windows, to the edges of the main window or outside of itom. Undocked script windows can be docked by releasing them above the edges of the main window. See :ref:`gui-editor-dock-drag-drop`.
