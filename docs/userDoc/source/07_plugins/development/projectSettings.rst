.. include:: ../../include/global.inc

.. sectionauthor:: Marc Gronle

Project settings for plugins
=======================================

Since you are building your plugins using **CMake** most of the following settings are automatically set. However, this document gives some hints about properties, which can maybe be helpful in case of any problems.

Prerequisites
----------------

For programming a plugin, you need at least the following things:

* A C++-compatible IDE. On Windows-machines it is recommended to program with *Visual Studio Professional 2010*, since |itom| is developed with this IDE, too. On Linux-machines you
  can for instance use the *QtCreator* or *Eclipse*. It is difficult to develop with *Visual Studio 2010 Express*, since you should install the *Qt AddIn for Visual Studio* in order to
  have a good support of |Qt| within *Visual Studio*. In the case you don't have the professional version of *Visual Studio*, better consider to use the *QtCreator* for Windows. You must have the |Qt| version installed, whose major and minor version number is equal than the version |itom| has been built with. Nevertheless the debugging of your plugin only is possible if you also have a debug-version of |itom| available on your computer, hence, you built it by yourself from the sources. Else, you only can test your plugin by extensively streaming debugging messages to the *std::cerr* or *std::cout* stream, which finally are displayed in the command line of |itom|.

* For running *itom*, you need *Python 3.2* installed on your machine.

* If you want to support the *itom*-internal *DataObject* (matrix structures), it is highly recommended to install *OpenCV2.3* or later on your machine.

Please consider to have all libraries, which you need, installed in the same version with respect to the processor type (32bit or 64bit).

General settings
-------------------

* compile your plugin as dynamic library (*dll* or *a* on linux).
* for the code generation use as runtime library the setting **Multithreaded-DLL (/Md)** or **Multithreaded-Debug-DLL** (/MDd) respectively.
* Don't use any precompiled headers on Windows.
* You can switch *openMP* on in order to support multi-processor calculations for parallelizable algorithms.
* Call your DLL "[yourName].dll" for the release-version and "[yourName]d.dll" for the debug-version.

Qt-dependent settings
---------------------

|itom| is written in *C++* using the |Qt|-framework. |Qt| provides platform-independent modules and classes which extend the possibilities of native C++. For example, |Qt| gives the
opportunity to build GUI-applications, have network and graphics support or to establish a platform-independent plugin system.

On the one hand, some functionalities of |Qt| can be used by the help of native Qt-applications, like the designer to build "what-you-see-is-what-you-get" user interfaces, the
translator to create translations of the application..., on the other hand C++ is enlarged by |Qt| by writing specific pre-compiler commands in the code. In both cases, these features
have to be translated into native C++-code during the pre-compiling process. Therefore the project files have to be adapted, such that the |Qt|-specific pre-steps will be triggered once
the project's compilation process is started. All this is done if you install the |Qt|-AddOn for Visual Studio (if developing with Visual Studio IDE).

Since the plugin, you will write, is based on *Qt*'s plugin system, these steps also have to be added to the plugin's pre-compiling steps. This can be realized by different ways:

#. You use the **QtCreator** as IDE and everything works fine (if the path to |Qt| is contained in the path variable and the environmental variable *QTDIR*)
#. You can use the professional version of Visual Studio together with the installed add-in **Qt Visual Studio Add-In**
#. You can use any other development environment and you have to add the necessary pre-compilation step by yourself in the appropriate project file.

The pre-processor-step contains the following steps:

#. In a folder "generated-files" additional files will be created for each class, containing the macro *Q_OBJECT* (moc-process).
#. Any user-interface file (.ui) will be transformed into an additional C++-class file, that is also contained in the "generated-files" folder (uic-process).
#. The translation tables will be created.
#. The resource-files will be parsed and an appropriate C++-file is created (rcc-process).

|Qt| is shipped with a number of different libraries (lying in the folder **$QTDIR$\bin**). You must link your application against the libraries, whose function you will need in your plugin.
It is always necessary to link against the library **QtCore** and **QtGui** if your plugin contains any user interface functionality. Other important libraries are
**QtOpenGL** for OpenGL-support, **QtSvg** for *Svg*-support or **QtXml**, **QtSql** or **QtNetwork**. For each of these libraries you plugin must have an entry in the *include*-directories and the *linker*-commands.

The pre-processor-definitions must contain the following entries:

* WIN32 or _WIN64
* QT_LARGEFILE_SUPPORT
* QT_PLUGIN
* QT_DLL
* QDESIGNER_EXPORT_WIDGETS

and for every Qt-library you need (in capital letters)

* QT_CORE_LIB
* QT_GUI_LIB
* ...

Include settings
------------------

Add the following include-directories ($(ITOM_QTDIR) is the path to the |Qt|-source directory - having subfolders like include or bin):

* .\GeneratedFiles
* .\GeneratedFiles\Debug or \Release
* $(ITOM_QTDIR)\include
* $(ITOM_QTDIR)\include\qtmain

and for every further Qt-library

* $(ITOM_QTDIR)\include\QtCore
* $(ITOM_QTDIR)\include\QtGui
* ...

Additionally you should add the path to OpenCV's include directory, if you are using or linking against the *DataObject*.

Linker settings
-----------------

Add as linker directory:

* The directory, where itom is saving the library to the dataObject.lib...
* The library-path of OpenCV
* The directory $(ITOM_QTDIR)\bin

Your plugin should at least link against the following libraries:

**DEBUG**:

* qtmaind.lib
* QtCored4.lib
* QtGuid4.lib
* ...further Qt libraries
* opencv_core$(ITOM_OPENCV_VER)d.lib
* DataObjectd.lib
* ...

**RELEASE**:

* qtmain.lib
* QtCore4.lib
* QtGui4.lib
* ...further Qt libraries
* opencv_core$(ITOM_OPENCV_VER).lib
* DataObject.lib
* ...

.. note::
    For more information about the deployment of plugins, including notes about the |Qt|-version compatibility, see `link to Qt-documentation <https://doc.qt.io/qt-5/deployment-plugins.html>`_
