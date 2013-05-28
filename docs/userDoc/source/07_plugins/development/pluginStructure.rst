.. include:: ../../include/global.inc

.. sectionauthor:: Marc Gronle

Basic programming structure of any itom-plugin
==============================================

Introduction
------------

The software |itom| obtains most functionality by mainly two concepts. On the one hand there is the python scripting language, which allows you to use almost all available python modules
that are available for python 3.2 or higher. On the other hand, the entire measurement system becomes powerful by the possibility to enhance its functionality by several plugins.

These plugins are separated into three main groups:

* Type **actuator**: Plugins of this basic type should be used if you want to connect any actuator, like motor stages, piezo actuators, focussing systems, ... to |itom| (see :ref:`plugin-class` and :ref:`plugin-class-actuator`)
* Type **dataIO**: Plugins of this basic type should be used for connecting any input or output device to |itom|. The main representative of this group are cameras as input device or the serial port as an input/ouput device (see :ref:`plugin-class` and :ref:`plugin-class-dataio`). This group is subdivided into the following sub-types:
    
    * **grabber** for cameras (Please consider that the class of the camera-plugin should not directly derive from *AddInDataIO* but from *AddInGrabber*, which is derived from the first.
    * **ADDA** for any analog-digital converters
    * **rawIO** for further input-output devices, like display windows for SLM or LCoS-sensors, serial ports or plugins which do not fit to any other group, since the type **dataIO** is the most flexibel plugin type.
    
* Type **algo**: Plugins of this type mainly contain different algorithms and/or advanced user interfaces like dialogs, main windows, widgets, ...  (see :ref:`plugin-class` and :ref:`plugin-class-algo`)

Each plugin is a different project in your programming environment and is finally compiled as shared library (DLL on Windows).


Plugin load mechanism of |itom|
-------------------------------

The |itom|-base directory contains a folder **plugins**. This folder itsself usually consists of different subfolders each having the name of a specific plugin. The folder can then
contain a release and/or debug-version of the specific plugin DLL as well as further files which are necessary for running the plugin. If your plugin is dependent on other files,
please consider to read the specific information about how to publish dependencies of each plugin.

At the startup of |itom|, the application recursively scans the **plugins** folder and looks for any *DLL*-file on Windows machines or *a*-file on a Linux operating system. Then each
DLL is tried to be loaded using the plugin system provided by the |Qt|-framework. The *DLL* can successfully be loaded if the following prerequisites are fullfilled:

* The plugin is a release version if |itom| is started in release mode OR
* The plugin is a debug version (this can for example be seen if the DLL-name ends with *...d.dll*) if |itom| is started in debug mode
* The plugin is compiled using the same major and minor version of |Qt| than |itom| (it is possible to load a plugin compiled with |Qt| 4.8.3 with |itom| compiled with 4.8.2)
* The plugin is compiled with the same compiler than |itom|
* If the plugin is dependent on other shared libraries which are not linked using a delay-load mechanism, the plugin can only be loaded if every necessary shared library can be found and successfully be loaded. If the dependency could not be loaded, the plugin-load fails with an error message *module could not be loaded*.
* The remarks contained in the plugin with respect to a minimum and maximum version number of |itom| must correspond to the version number of your |itom|
* The plugin must be compiled with the same version string of the class **ito::AddInInterface** than the version contained in |itom| (this is not the general version of |itom|). The version string of **AddInInterface** can be seen at the end of the file **addInInterface.h** in the **common**-folder.

An overview about the load status of all detected library files can be seen by calling the dialog **loaded plugins**, accessible by |itom|'s menu **help >> loaded plugins...**.

Finally, every successfully loaded plugin is included in the dock-widget **Plugin** of |itom|.

Basic plugin structure
----------------------

.. highlight:: c

Every plugin consists at least of two classes, which are both derived from two different base classes. All possible base classes are provided in the files::

    addInInterface.h
    addInInterface.cpp

which are contained in the folder **common** of |itom|'s SDK. This folder contains further files which can be used in every plugin and contain interfaces and helper libaries with usefull functions for successfully and easily program a plugin. Additionally |itom| provides an application programming interface (API) such that plugins can access important methods of |itom|. For more information see :ref:`plugin-itomAPI`.

The two classes of the plugin are as follows:

1. Interface- or factory-class (derived from class **AddInInterfaceBase**)

    This class must be derived from the class **addInInterfaceBase** and is the communcation tunnel between |itom| and the plugin itself using the plugin-framework of |Qt|. The plugin framework creates one single instances of this class when the plugin DLL is loaded (that means at startup of |itom|). Therefore this class is considered to be a singleton instance and since it is always loaded by |itom| even if it is not really needed, this class is kept small and only provides basic information about the plugin itself.
    
    For further information about the structure of this interface class see :ref:`Plugin Interface Class <plugin-interface-class>`.

2. Individual plugin class (derived from class **AddInDataIO**, **AddInGrabber**, **AddInActuator** or **AddInAlgo**)

    This class is the main class of the plugin and should contain the main functionality of the plugin. Depending on the plugin type, this class is derived from any of the classes **AddInDataIO**, **AddInGrabber**, **AddInActuator** or **AddInAlgo**, which are also contained in the files mentioned above. All this classes internaly are derived from the base class **AddInBase**, which is the most general class used for plugin handling and organization in |itom|. Please do not directly derive from **AddInBase**.
    
    In the case of an actuator, a camera or any other IO-device, every opened device is represented by one individual instance of its corresponding plugin class. Hence, it is possible to have multiple instances of every class opened in |itom|. The creation and deletion of any instance is at first requested by the **AddInManager** class (an internal class of |itom|) which itself redirects this request to the singleton instance of the interface class in the corresponding plugin (This is the interface class mentioned in point 1 above).
    
    In the case of an algorithm-plugin, this class mainly contains a set of static methods, each being one individual algorithm or user interface. At startup of |itom| the singleton instance of the interface-class is created. Additionally, this individual plugin class also is instantiated once (singleton) at startup of itom and its internal *init*-method provides an overview (list) of all available algorithm and user-interface functions to |itom|. Additionally the default parameter sets for all algorithms and widget-methods are requested by |itom| and startup and are then cached in order to provide faster access in any subsequent function calls.
    
    Further information about the common parts of the plugin class, independent on the plugin's type, see :ref:`plugin-class`. For detailed information about the implementation of the different plugin types, see :ref:`plugin-class-dataio`, :ref:`plugin-class-actuator` or :ref:`plugin-class-algo`.
    
Communication between |itom|, Python and each plugin
----------------------------------------------------

The communcation to plugins of type **actuator** and **dataIO** is only possible by calling the public methods defined in the base classes **AddInActuator** or **AddInDataIO**. In Python, there exist two classes **dataIO** and **actuator**. Both have an interface that is analog to the corresponding interface **AddInActuator** or **AddInDataIO** in C++. Therefore, if a certain method of these classes is called in python, the call is redirected to the corresponding plugin-method. However, this call is executed across a thread-change, since both python and each plugin (besides the algorith-plugins) "live" in their own thread.
