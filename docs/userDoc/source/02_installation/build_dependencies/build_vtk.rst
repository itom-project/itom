:orphan:

.. include:: ../../include/global.inc

.. _build_dependencies_vtk:

Compile VTK 9
=============

Unzip the VTK source on your hard drive. Create a build_x64/build_x86 folder 
and execute than CMake with this two folders. 

* Change **CMAKE_INSTALL_PREFIX** to **${MAINDIR}/3rdPartyPCL/vtk9.2.2**
* Set **Qt_Prefix_DIR** to to **${MAINDIR}/3rdParty/Qt6.4.1/msvc2016_64/lib/cmake/Qt6**. 
* Set **VTK_MODULE_ENABLE_VTK_GUISupportQt** to the value **YES**.
* Set **VTK_MODULE_ENABLE_VTK_GUISupportQtSql** to the value **YES**.
* Set **VTK_MODULE_ENABLE_VTK_RenderingOpenGL2** to the value **YES**.
* Set **VTK_MODULE_ENABLE_VTK_RenderingQt** to the value **YES**.
* Set **VTK_MODULE_ENABLE_VTK_ViewsQt** to the value **YES**.
* Set **VTK_MODULE_ENABLE_VTK_WrappingTools** to the value **YES**.

Compile VTK 8
==============

Unzip the VTK source on your hard drive. Create a build_x64/build_x86 folder 
and execute than CMake with this two folders. 

* Uncheck **BUILD_EXAMPLES, BUILD_TESTING, HDF5_USE_FOLDERS, HDF5_EMBEEDDED_LIBINFO**
* Check **BUILD_SHARED_LIBS**
* Check **Module_vtkGUISupportQt, Module_vtkGUISupportQtOpenGL, Module_vtkGUISupportQtSQL, 
  Module_vtkRenderingQT and Module_vtkViewsQt**. 
* Add a new entry: name = **"CMAKE_DEBUG_POSTFIX"**, type = **"STRING"** with the value = **"-gd"** for version <9
* Add a new entry: name = **"VTK_USE_QT"**, type = **"BOOL"** with the value = checked for version <9
* Add a new entry: name = **"VTK_USE_GUISUPPORT"**, type = **"BOOL"** with the value = checked for version <9
* Change **CMAKE_INSTALL_PREFIX** to **${MAINDIR}/3rdPartyPCL/vtk8.2.0**
* If an error occures with wrong Qt Version, change **VTK_QT_VERSION** to **5**
* Choose with the variable **VTK_RENDERING_BACKEND** which OpenGL is used for VTK/PCL. 
* Set the entry **Qt_Prefix_DIR** to to **${MAINDIR}/3rdParty/Qt5.12.1/msvc2017_64/lib/cmake/Qt5**. 
* Check **VTK_BUILD_QT_DESIGNER_PLUGIN**.

.. note::

    Check the Entries **Qt_Prefix_DIR**, **QtCore_DIR**, **QtSql_DIR**, ..., 
    if they are set to the right path. 

.. warning::

    1. Before starting the compilation open in the folder **VTK\\build\\GUISupport\\Qt** 
        the **PluginInstall.cmake** file and change in line **5** **"QVTKWidgetPlugin.dll"** 
        to **"QVTKWidgetPlugin-gd.dll"**
    2. Start **DEBUG** compilation in Visual Studio
    3. Change the **"QVTKWidgetPlugin-gd.dll"** back to **"QVTKWidgetPlugin.dll"** 
        and start **RELEASE** compilation
    
.. warning:: 

    Disable deprecation warnings by setting the cmake variable: **VTK_LEGACY_SILENT:ON**

