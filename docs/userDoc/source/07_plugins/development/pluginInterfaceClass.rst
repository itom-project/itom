.. include:: ../../include/global.inc

.. sectionauthor:: Marc Gronle

.. _plugin-interface-class:

Plugin interface class
========================

Every |itom| plugin must consist of at least two classes. One class is the real plugin class, that represents one device. Therefore it is possible to create multiple instances of this class, hence, open multiple devices of the same plugin. The other class is a necessary structure in order to allow the communication between |itom| and the plugin. This is the so-called interface class, inherited from the abstract base class **AddInInterfaceBase**.

This base class is part of the **ito**-namespace and is defined in the file::

    addInInterface.h

This file is located in the *common* directory of the include directory of the |itom| SDK. You need to link against the library **itomCommonQtLib** in the SDK.

In the main header file of your plugin (with the exemplary name **MyPlugin**), use the following demo code in order to create that class:

.. code-block:: c++
    :linenos:

    //myPlugin.h

    #include "common/addInInterface.h" //adapt the path depending on the location of your plugin

    class MyPluginInterface : public ito::AddInInterfaceBase
    {
        Q_OBJECT
        Q_INTERFACES(ito::AddInInterfaceBase)
        PLUGIN_ITOM_API

        public:
            MyPluginInterface(QObject *parent = 0);
            ~MyPluginInterface() {};
            ito::RetVal getAddInInst(ito::AddInBase **addInInst);

        private:
            ito::RetVal closeThisInst(ito::AddInBase **addInInst);
    };

In the code example above, the macro directives *Q_OBJECT* and *Q_INTERFACES(ito::AddInInterfaceBase)* (lines 7+8) force the compiler in the pre-compilation step to create the necessary code (done by the |Qt| framework) such that the class fits to the |Qt|-plugin system and is able to communicate by the common signal-slot-system of |Qt|. Remember that every class which is finally derived from the *QObject*-class (like *AddInInterfaceBase* is, too) must have the *Q_OBJECT* macro defined.

The constructor *MyPluginInterface(...)*, defined in line 11 is called once by the AddIn-Manager of |itom| at startup in order to create the singleton instance of the class *MyPluginInterface*. In the body of this method you should provide basic information about your plugin (see section :ref:`plugin-interface-class-constructor`).

The destructor in line 12 usually does not require further implementation, such that the empty body already can be given in the header file.

Finally, there are also the methods *getAddInInst* and *closeThisInst* which are the most important methods. If an user or some other part of |itom| request an instance of this plugin (that means not an instance of the interface we are talking in this section, but of the real plugin), the AddInManager of |itom| calls the method *getAddInInst* of the corresponding interface class. Then this interface has to create an instance of the plugin and set the given double-pointer parameter to the pointer of this newly created instance.

Inversely, the AddInManager of |itom| will call *closeThisInst* of an interface in order to force the plugin interface class to delete the plugin instance, given by the *addInInst* parameter. This mechanism is usually used by so-called factory-classes. Therefore we can consider the interface class to be a factory for one or more instances of the plugin itself (For information about the plugin class see :ref:`plugin-class`).

.. _plugin-interface-class-constructor:

The constructor of the plugin interface class
---------------------------------------------

In your main source file of your plugin you can implement the constructor of the plugin interface class in the following exemplary way:

.. code-block:: c++

    MyPluginInterface::MyPluginInterface(QObject *parent)
    {
        m_type          = ito::typeActuator; //or: ito::typeAlgo, ito::typeDataIO, ito::typeDataIO | ito::typeGrabber ...
        setObjectName("MyPlugin"); //this is the name of the plugin how it appears in itom

        m_description   = QObject::tr("Description of MyPlugin");
        m_author        = "Author's name";
        m_version       = CREATEVERSION(0,1,0);
        m_minItomVer    = CREATEVERSION(1,0,0);
        m_maxItomVer    = MAXVERSION;

        m_autoLoadPolicy = ito::autoLoadKeywordDefined;
        m_autoSavePolicy = ito::autoSaveAlways;

        //initialize mandatory parameters for creating an instance of MyPlugin
        m_initParamsMand.append( ito::Param("param1", ito::Param::String, \
            "defaultValue", tr("translatable description").toLatin1().data()) );
        ...

        //initialize optional parameters for creating an instance of MyPlugin
        m_initParamsOpt.append( ito::Param("optParam1", ito::Param::Int, 0, 10, 5, \
           tr("translatable description of optParam1").toLatin1().data()) )
        ...
    }

At first, the constructor consists of a section where you define basic information about the plugin itself. In the second part you will define a list of mandatory and optional parameters which are required if any user wants to create an instance of the plugin, e.g. the user wants to open a new camera or connect any motor.

**Part 1 (Basic information):**

.. c:member:: int m_type

    Type of this plugin. Possible types are an OR-combination of the enumeration **ito::tPluginType**:

    * typeActuator for actuator-plugins
    * typeDataIO | typeGrabber for cameras and other grabbing devices
    * typeDataIO | typeADDA for any analog/digital converters
    * typeDataIO | typeRawIO for any other input-output-devices, like serial ports, display windows...
    * typeAlgo for a plugin providing algorithms, filters or any other methods as well as graphical user interfaces, dialogs, ... which enhance the functionality of |itom|

.. c:member:: void setObjectName(const QString &name)

    use this method to set the name of your plugin. This name should be simple and should not contain special characters, since it not only appears in the list of plugins but is also the string used for initializing a plugin by the python scripting language.

.. c:member:: QString m_description

    Give an advanced description of your plugin.

.. c:member:: QString m_author

    Use this string to denote the author(s) of this plugin

.. c:member::  int m_version

    This integer variable contains the version of your plugin. A version string always consists of a major, minor and patch value. All these values are combined in the integer variable and can be created using the macro **CREATEVERSION(major,minor,patch)** (defined in *sharedStructures.h*), where the values major, minor and patch are integer values, too.

.. c:member::  int m_minItomVer

    Use this variable to denote the minimum version number of |itom| which is necessary to run this plugin. If you don't have any specific minimum version, use the macro **MINVERSION**, defined in *sharedStructures.h* (folder *common*).

.. c:member::  int m_maxItomVer

    Use this variable to denote the maximum version number of |itom|. Versions higher than this value do not allow to run this plugin. If you don't care about any maximum version, use the macro **MAXVERSION**, defined in *sharedStructures.h* (folder *common*).

.. c:member:: ito::tAutoLoadPolicy m_autoLoadPolicy

    Depending on the value of this variable, the internal parameters of the plugin can be loaded from a *xml*-file and set after the plugin's *init*-method has been called. The possible values for that variable are given by the enumeration **ito::tAutoLoadPolicy** and are

    .. code-block:: c++

        enum tAutoLoadPolicy {
            autoLoadAlways           = 0x1, /*!< always loads xml file by addInManager */
            autoLoadNever            = 0x2, /*!< never automatically loads parameters from xml-file (default) */
            autoLoadKeywordDefined   = 0x4  /*!< only loads parameters if keyword autoLoadParams=1 exists in python-constructor */
        };

    For more information about the loading and/or saving of plugin's parameters, see :ref:`plugin-autoloadsave-policy`.

.. c:member:: ito::tAutoSavePolicy m_autoSavePolicy

    Depending on the value of this variable, the internal parameters of the plugin can be saved to a *xml*-file at shutdown of a plugin instance. The possible values for that variable are given by the enumeration **ito::tAutoSavePolicy** and are

    .. code-block:: c++

        enum tAutoSavePolicy {
            autoSaveAlways          = 0x1, /*!< always saves parameters to xml-file at shutdown */
            autoSaveNever           = 0x2  /*!< never saves parameters to xml-file at shutdown (default) */
        };

    For more information about the loading and/or saving of plugin's parameters, see :ref:`plugin-autoloadsave-policy`.

.. c:member:: bool m_callInitInNewThread

    Usually, the plugin's init method, where for instance the hardware is started and initialized, is called in a new thread in order to keep the GUI reactive during the whole process. If you change this member from
    its default value **true** to **false**, **init** is executed in the main thread and afterwards the plugin is moved to the new thread. For more information, see :ref:`plugin-class` or the :ref:`info box <plugin-class-callInitThread>`.

**Part 2 (mandatory and optional parameters):**

.. note::
    This part is only important if you build plugins of the basic types **dataIO** or **actuator**, since only plugins of these types can have multiple parameters, hence, it is useful to parametrize their constructors. For algorithm- or filter-plugins, you can let the vectors **m_initParamsMand** and **m_initParamsOpt** unchanged (hence empty).

If you create an instance of a plugin using the python language, you have mainly two possibilities:

* Plugins of type **dataIO** are addressed using the python type **dataIO**, which is a class of the module |pyItom|:

    .. code-block:: python

        from itom import * # usually this import already has been done for you
        variable = dataIO(PluginName,mandatoryParam1, ..., mandatoryParamN, optionalParam1, ..., optionalParamN)

    OR

    .. code-block:: python

        import itom
        variable = itom.dataIO(PluginName, mandatoryParam1, ..., optionalParam1, ...)

* Plugins of type **actuator* are addressed using the python type **actuator**, which is a class of the module |pyItom|, too. The call is then analogous to the examples above:

    .. code-block:: python

        variable = actuator(PluginName,mandatoryParam1, ..., mandatoryParamN, optionalParam1, ..., optionalParamN)

* Plugins of type **algo** do not have any corresponding class in |pyItom|, since they are globally organized by |itom|. Algorithms can be called using the method :py:func:`filter`, windows, dialogs or further user interfaces provided by plugins are loaded using the static method :py:func:`createNewPluginWidget` of class :py:class:`itom.uiDialog`.

The constructor of each plugin can have a list of mandatory and optional parameters, which must or can be provided if creating an instance of the plugin. Internally, each parameter is a value of type **Param**, which is a class of |itom| and provides values of different types. Each value has a specific name, a default value and a description string, which should be given or set to NULL. Additionally, depending on the parameters type, a minimum and maximum value can be indicated. For more information about class *Param* see :ref:`plugin-Params`.

The mandatory parameters are contained in the vector

.. code-block:: c++

    QVector<ito::Param> m_initParamsMand

Using the methods *append* or *insert* you can add an arbitrary number of values (type *Param*) to this vector. The type *QVector* is a |Qt|-specific class which is similar to *std::vector*. The optional parameters are analogously contained in the vector

.. code-block:: c++

    QVector<ito::Param> m_initParamsOpt

If one is creating an instance of the plugins, e.g. using the python commands above, |itom| is reading the given vector of mandatory of optional parameters. The first parameter of the constructors of the python class :py:class:`itom.dataIO` or :py:class:`itom.actuator` stands for the name of the plugin. The number of the following parameters must be equal or bigger than the length of the mandatory parameter vector. The first *n* parameters must exactly fit to the type, order and possible boundary values of the mandatory parameter vector. This vector is then copied and the values are replaced by the values given by the python-constructors.

If the following parameters in the constructor don't have any keywords, they must also fit to the types,... of the optional parameter vector. If there are not enough parameters given, the default value will be taken. Additionally, if the user gives keywords to the parameters, each parameter will be checked against its corresponding value in the optional parameter vector where keyword and parameter-name are equal. After the first parameter having a keyword no keyword-less parameters are accepted.

This is an example of creating a plugin with a set of parameters, where the last two parameters are tagged with their keywords:

.. code-block:: python

    variable = dataIO("MyPlugin",2.0,"test",delay=1000,file="C:\\test.dat"")

After that the mandatory and optional parameter vectors are read, copied and that their values are replaced by the values given by the constructor, the instance of the plugin is created and the method **init** of the plugin class is called with the mandatory and optional parameter vector as argument. That's the basic way such a plugin instance is created and initialized.

The set of mandatory and optional parameters of each plugin, including their default, minimum and maximum value, their name and description string, can be returned in python using the method :py:func:`itom.pluginHelp`.

.. _plugin-interface-class-getAddInInst:

Method *getAddInInst* of the plugin interface class
---------------------------------------------------

As default implementation, you can copy the following code block for your implementation of the *getAddInInst*-method:

.. code-block:: c++
    :linenos:

    ito::RetVal MyPluginInterface::getAddInInst(ito::AddInBase **addInInst)
    {
        NEW_PLUGININSTANCE(MyPlugin)
        return ito::retOk;
    }

In case of an algorithm plugin use:

.. code-block:: c++
    :linenos:

    ito::RetVal MyPluginInterface::getAddInInst(ito::AddInBase **addInInst)
    {
        NEW_PLUGININSTANCE(MyPlugin)
        REGISTER_FILTERS_AND_WIDGETS
        return ito::retOk;
    }

Since your plugin instance (**MyPlugin**) is finally derived from **AddInBase**, its private member **m_uniqueID** is automatically given an auto-incremented, unique number. Additionally if possible assign a string identifier that helps to identify the opened device. Set the identifier using the method **setIdentifier**. Please only set the identifier in the constructor or in the **init**-method of the plugin itself. The identifier is saved in the member **m_identifier**.

In the method above, it is assumed that your main class of your plugin *MyPlugin* is called *MyPlugin*, too. Then in line 3, a new instance of that class is created and this new instance is noticed about its own factory class in line 4. The factory class is hereby the pointer to this singleton interface class instance. Finally the given double pointer is set to the pointer of the newly created plugin instance. Finally, every plugin interface class has a protected member vector called *m_InstList* which contains a list of plugin instances opened by this interface (or factory). The newly created plugin is added to this list in line 6.

The return value of this method is of type **ito::RetVal**, which is set to the status **Ok**. For more information about the return value class **ito::RetVal** see :ref:`plugin-retVal`.


.. _plugin-interface-class-closeThisInst:

Method *closeThisInst* of the plugin interface class
----------------------------------------------------

For this method, you can basically copy the following default implementation:

.. code-block:: c++
    :linenos:

    ito::RetVal MyPluginInterface::closeThisInst(ito::AddInBase **addInInst)
    {
        REMOVE_PLUGININSTANCE(MyPlugin)
        return ito::retOk;
    }

The AddInManager of |itom| is calling this method if the given plugin instance (parameter **addInInst**) should be deleted. If the parameter pointer is available, the plugin instance is removed from the list of loaded plugin instances (see :ref:`plugin-interface-class-getAddInInst`) and the plugin instance is deleted.
