.. include:: ../../include/global.inc

.. sectionauthor:: Marc Gronle

.. _plugin-externalLibraries:

Link or load external libraries
===============================

In your plugin, you have different possibilities to use external or 3rd party libraries:

Link to a static library
------------------------

If you add an appropriate entry to the linker settings of your project file, it is possible to link to external, static libraries.
This is for example the case when linking to the **dataObject.lib** or **dataObject.so** (Linux) or if using any components
of the *PointCloudLibrary*. When compiling your plugin, the whole implementation
of the library is compiled into your plugin-library and you don't need to distribute the external file.

Link to a shared library
------------------------

You can also link to an external shared library by adding the corresponding entry to the linker settings, too. This is for example the case
when using *OpenCV*-methods, which use implicitly do, if you link against the dataObject. Then you must add the corresponding **.lib** or
**.so** files to your linker-settings and provide the corresponding **.dll** or **.a** file (in debug or release, if possible). Then, you have to take
care that |itom| is able to find the external library. Please consider, that this external library file is not detected relatively to the location of your
plugin but relatively to the executable of |itom| itself. Therefore, you have these possibilities to distribute the external library file:

* Add its containing folder to the path variable of your operating system.
* Add it to any path which already is contained in the path variable of your OS (e.g. **system32** on Windows).
* Directly add it to the |itom|-folder (not recommended, since leads to "crazy" folder structure)
* Directly add it to the folder **lib** contained in the |itom|-folder. The *lib**-folder is added to the path variables that are passed to |itom| at startup. This is a good possibility to provide the external library file, however it can also lead to conflicts with other plugins, that need the same external library, however in an other version. Therefore check whether you can share the same version with other plugins. In the default implementation of |itom|, there is also some default libraries of **Glut**, **FFTW**... that should be used.
* Try to indicate the shared library as delay-loaded module. Than you also adapt the path variable of your plugin to a folder of your choice, before the plugins tries to load the shared library. This is a conflict-free way how to access shared libraries. Let's make an example: Your plugin **MyPlugin** lies after compilation in the folder **plugin/myPlugin**, that is a subfolder of |itom|. Then put your external shared library file in the subfolder **lib** of **plugin/myPlugin**. Then add this path to the current path variable of the application by adding the following code for instance in the constructor or the **init**-method of your plugin implementation::

    // Get the path to the plugin directory
    QDir dllDir = QCoreApplication::applicationDirPath();
    if( !dllDir.cd("plugins/MyPlugin") )
    {
        dllDir.cdUp();
        dllDir.cd("plugins/MyPlugin");
    }
    dllDir.cd("lib"); //move to lib folder
    QString dllDir2 = QDir::cleanPath(dllDir.filePath(""));

    // Add the plugin path to the path environment variable
    char *oldpath = getenv("path");
    char *newpath = (char*)malloc(strlen(oldpath) + dllDir2.size() + 10);
    newpath[0] = 0;
    strcat(newpath, "path=");
    strcat(newpath, dllDir2.toLatin1().data());
    strcat(newpath, ";");
    strcat(newpath, oldpath);
    _putenv(newpath);
    free(newpath);

.. note::

    The path variable of your operating system is always copied and then passed to a newly started application. Therefore you can adapt this copy without influencing the overall path environment variable.

Load external library at runtime
--------------------------------

The most complicated way to access an external library with respect to the programming cost is to use the command **LoadLibrary** or the platform-independent |Qt|-class **QLibrary** in order load an external library at runtime of your plugin. Then you need to resolve the symbols in the library in order to access them afterwards in a function-call. The advantage of this
method however is, that the library can be at any location since you are able to load the library with its absolute filename. See the |Qt| documentation for details about the class **QLibrary**, that is recommended to use.

.. note::

    If you link to external libraries, please consider always the license requirements of the external library.
