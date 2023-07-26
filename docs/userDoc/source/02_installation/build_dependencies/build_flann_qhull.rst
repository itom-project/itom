:orphan:

.. include:: ../../include/global.inc

.. _build_dependencies_flann_qhull:

Compile Flann, QHull
========================

If you compile Flann or QHull by yourself, add the entry **CMAKE_DEBUG_POSTFIX**
of type **STRING** with the value = **"-gd"**. * Uncheck in CMake **BUILD_DOC,
BUILD_EXAMPLES, BUILD_EXAMPLES, BUILD_MATLAB_BINDINGS, BUILD_PYTHON_BINDINGS,
BUILD_TESTS**.

* Set the **CMAKE_INSTALL_PREFIX** to the Flann or QHull folder in the
  **${MAINDIR}/3rdPartyPCL** folder.
* Compile the **INSTALL** solution of the MSVC Project.
