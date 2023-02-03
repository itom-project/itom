:orphan:

.. include:: ../../include/global.inc

.. _build_dependencies_pcl:

Compile PCL (QT6, VTK9)
=======================

Unpack the PCL source on your hard drive. Create a build_x64/build_x86 folder and 
execute CMake with this folders. After each step, you must start the CMake configuration, 
otherwise the new entries will not be added. 

* Set **EIGEN_INCLUDE_DIR** to ${MAINDIR}/3rdPartyPCL/Eigen3.4.0
* Set **FLANN_INCLUDE_DIR** to ${MAINDIR}/3rdPartyPCL/flann1.9.1/include
* Set **FLANN_LIBRARY** to ${MAINDIR}/3rdPartyPCL/flann1.9.1/lib/flann_cpp_s.lib
* set **FLANN_LIBRARAY_DEBUG** to ${MAINDIR}/3rdPartyPCL/flann1.9.1/lib/flann_cpp_s-gd.lib
* Set **Boost_INCLUDE_DIR** to ${MAINDIR}/3rdPartyPCL/boost1.81.0 (Boost library must be named libboost... \*.lib) 
* Add new path entry: **Boost_LIBRARY_DIR** with path ${MAINDIR}/3rdPartyPCL/boost1.69.0/lib64-msvc-14.3
* Add new bool entry: **Boost_USE_STATIC_LIBS** with value **ON**
* Set **QHULL_INCLUDE_DIR** include dir to ${MAINDIR}/3rdPartyPCL/qhull-2015.2/include
* Set **QHULL_LIBRARY** dir to ${MAINDIR}/3rdPartyPCL/qhull-2015.2/lib/qhullstatic.lib
* Set **QHULL_LIBRARY_DEBURG** dir to ${MAINDIR}/3rdPartyPCL/qhull-2015.2/lib/qhullstatic_d.lib
* Set **Qt6_DIR** to to **${MAINDIR}/3rdParty/Qt6.4.1/msvc2016_64/lib/cmake/Qt6**. 
* Set **CMAKE_INSTALL_PREFIX** ${MAINDIR}/3rdPartyPCL/pclM.M.P


Compile PCL (QT5, VTK8) 
=======================

Unpack the PCL source on your hard drive. Create a build_x64/build_x86 folder and 
execute CMake with this folders. After each step, you must start the CMake configuration, 
otherwise the new entries will not be added. 

* Set **EIGEN_INCLUDE_DIR** to ${MAINDIR}/3rdPartyPCL/Eigen3.3.7
* Set **FLANN_INCLUDE_DIR** to ${MAINDIR}/3rdPartyPCL/flann1.9.1/include
* Set **FLANN_LIBRARY** to ${MAINDIR}/3rdPartyPCL/flann1.9.1/lib/flann_cpp_s.lib
* set **FLANN_LIBRARAY_DEBUG** to ${MAINDIR}/3rdPartyPCL/flann1.9.1/lib/flann_cpp_s-gd.lib
* Set **Boost_INCLUDE_DIR** to ${MAINDIR}/3rdPartyPCL/boost1.69.0 (Boost library must be named libboost... \*.lib) 
* Add new entry: **Boost_LIBRARY_DIR** with path ${MAINDIR}/3rdPartyPCL/boost1.69.0/lib64-msvc-14.0
* Set **QHULL_INCLUDE_DIR** include dir to ${MAINDIR}/3rdPartyPCL/qhull-2015.2/include
* Set **QHULL_LIBRARY** dir to ${MAINDIR}/3rdPartyPCL/qhull-2015.2/lib/qhullstatic.lib
* Set **QHULL_LIBRARY_DEBURG** dir to ${MAINDIR}/3rdPartyPCL/qhull-2015.2/lib/qhullstatic_d.lib
* Set **VTK_DIR** to ${MAINDIR}/3rdPartyPCL/vtk8.2.0/lib/cmake/vtk-8.0
* Add new entry: **QVTK_LIBRARY_DEBUG** with **FILEPATH** ${MAINDIR}/3rdPartyPCL/vtk8.2.0/lib/vtkGUISupportQtOpenGL-8.2-gd.lib
* Add new entry: **QVTK_LIBRARY_RELEASE** with **FILEPATH** ${MAINDIR}/3rdPartyPCL/vtk8.2.0/lib/vtkGUISupportQtOpenGL-8.2.lib
* Set **Qt5Concurrent_DIR** to ${MAINDIR}/3rdParty/Qt5.12.1/5.12/msvc2017_64/lib/cmake/Qt5Concurrent
* Set **Qt5OpenGl_DIR** to ${MAINDIR}/3rdParty/Qt5.12.1/5.12/msvc2017_64/lib/cmake/Qt5OpenGl_DIR
* Set **Qt5Widgets_DIR** to ${MAINDIR}/3rdParty/Qt5.12.1/5.12/msvc2017_64/lib/cmake/Qt5Widgets_DIR
* check **BUILD_surface_on_nurbs** and **BUILD_visualization**
* uncheck **BUILD_global_tests**, **BUILD_examples**, **BUILD_apps**, **BUILD_simulation**
* uncheck **PCL_ENABLE_AVX** for version >1.12.0
* Set **CMAKE_INSTALL_PREFIX** ${MAINDIR}/3rdPartyPCL/pclM.M.P

* Check **PCL_SHARED_LIB**
* Delete **PCL_BUILD_WITH_BOOST_DYNAMIC_LINKING_WIN32**
* Delete **PCL_BUILD_WITH_FLANN_DYNAMIC_LINKING_WIN32**
* Add **BOOST_USE_STATIC**
* Add **FLANN_USE_STATIC**
* Add **QHULL_USE_STATIC**


.. note::
    
    The created *exe*-files are not needed to run |itom|. Delete all exe-files 
    in the *${MAINDIR}/3rdPartyPCL/pcl1.9.1/bin* folder. 
    
.. warning::
    
    PCL version 1.8.0 causes a compilation error due to some syntax error. 
    A workaround can be find here: https://stackoverflow.com/questions/38508319/pcl-visualizer-cpp-vs-2015-build-error/

.. warning::

    In the case of a CMake Error: **Requested modules not available: vtkGUISupportQtWebkit**
    Delete the VTK_MODULE **vtkGUISupportQtWebkit** in the 
    **VTK_INSTALL_DIR\\lib\\cmake\\vtk-8.2\\VTKConfig.cmake** in line 118: **set(VTK_MODULES_ENABLED "...")**
