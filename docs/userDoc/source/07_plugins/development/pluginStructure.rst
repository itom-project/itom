.. include:: ../../include/global.inc

.. sectionauthor:: Marc Gronle

Introduction to plugins
==============================================

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

For information about loading plugins in |itom|, see the section :ref:plugins-loading.

Basic plugin structure
----------------------

.. highlight:: c

Every plugin consists at least of two classes, which are both derived from two different base classes. All possible base classes are provided in the file::

    addInInterface.h

which is contained in the folder **include/common** of |itom|'s SDK. This folder contains further header files which can be used in every plugin and contain interfaces and helper libaries with useful functions for successfully and easily program a plugin. For using these files you need to link your plugin agains the libraries **itomCommonLib** and **itomCommonQtLib**. Additionally |itom| provides an application programming interface (API) such that plugins can access important methods of |itom|. For more information see :ref:`plugin-itomAPI`.

The two classes of the plugin are as follows:

1. Interface- or factory-class (derived from class **AddInInterfaceBase**)

    This class must be derived from the class **AddInInterfaceBase** and is the communication tunnel between |itom| and the plugin itself using the plugin-framework of |Qt|. The plugin framework creates one single instance of this class when the plugin DLL is loaded (that means at startup of |itom|). Therefore this class is considered to be a singleton instance and since it is always loaded by |itom| even if it is not really needed, this class is kept small and only provides basic information about the plugin itself.
    
    For further information about the structure of this interface class see :ref:`Plugin Interface Class <plugin-interface-class>`.

2. Individual plugin class (derived from class **AddInDataIO**, **AddInGrabber**, **AddInActuator** or **AddInAlgo**)

    This class is the main class of the plugin and should contain the main functionality of the plugin. Depending on the plugin type, this class is derived from any of the classes **AddInDataIO**, **AddInGrabber**, **AddInActuator** or **AddInAlgo**, which are also contained in the files mentioned above. All this classes internally are derived from the base class **AddInBase**, which is the most general class used for plugin handling and organization in |itom|. Please do not directly derive from **AddInBase**.
    
    In the case of an actuator, a camera or any other IO-device, every opened device is represented by one individual instance of its corresponding plugin class. Hence, it is possible to have multiple instances of every class opened in |itom|. The creation and deletion of any instance is at first requested by the **AddInManager** class (an internal class of |itom|) which itself redirects this request to the singleton instance of the interface class in the corresponding plugin (This is the interface class mentioned in point 1 above).
    
    In the case of an algorithm-plugin, this class mainly contains a set of static methods, each being one individual algorithm or user interface. At startup of |itom| the singleton instance of the interface-class is created. Additionally, this individual plugin class also is instantiated once (singleton) at startup of itom and its internal *init*-method provides an overview (list) of all available algorithm and user-interface functions to |itom|. Additionally the default parameter sets for all algorithms and widget-methods are requested by |itom| and startup and are then cached in order to provide faster access in any subsequent function calls.
    
    Further information about the common parts of the plugin class, independent on the plugin's type, see :ref:`plugin-class`. For detailed information about the implementation of the different plugin types, see :ref:`plugin-class-dataio`, :ref:`plugin-class-actuator` or :ref:`plugin-class-algo`.
    
Communication between itom, Python and each plugin
----------------------------------------------------

The communication to plugins of type **actuator** and **dataIO** is only possible by calling the public methods defined in the base classes **AddInActuator** or **AddInDataIO**. In Python, there exist two classes **dataIO** and **actuator**. Both have an interface that is analog to the corresponding interface **AddInActuator** or **AddInDataIO** in C++. Therefore, if a certain method of these classes is called in python, the call is redirected to the corresponding plugin-method. However, this call is executed across a thread-change, since both python and each plugin (besides the algorith-plugins) "live" in their own thread.
