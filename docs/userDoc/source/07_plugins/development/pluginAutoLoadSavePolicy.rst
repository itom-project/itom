.. include:: ../../include/global.inc

.. sectionauthor:: Marc Gronle

.. _plugin-autoloadsave-policy:

Automatic loading and saving of plugin parameters
=================================================

|itom| has the optional ability, that all plugin parameters (type *Param*), which are part of the **m_params** map of the plugin-class and do not have the *typeNoAutoSave*-flag defined, can be stored in a plugin-specific xml-file when the plugin instance is closed. This saving is not only dependent on the plugin but also on its unique identifier.
|itom| has the optional ability, that all plugin parameters (of type **Param** or **ParamBase**), which are part of the **m_params** map of the plugin-class and do not have the *typeNoAutoSave*-flag defined, can be stored in a plugin-specific xml-file when the plugin instance is closed. This saving is not only dependent on the plugin but also on its unique identifier.

Additionally, the decision whether these parameters should be saved or not, is set by the member variable **m_autoSavePolicy** of the interface class of the plugin (see :ref:`plugin-interface-class`). This variable can have the following values:

.. code-block:: c++

    enum tAutoSavePolicy {
        autoSaveAlways          = 0x1, /*!< always saves parameters to xml-file at shutdown */
        autoSaveNever           = 0x2  /*!< never saves parameters to xml-file at shutdown (default) */
    };

If the member has the value *autoSaveAlways*, all parameters contained in the map *m_params* of the plugin instance are saved in a xml-subtree. This subtree is dependent on the unique identifier of the plugin. Remember, that only plugins which do not have the flag *typeNoAutoSave* will be saved (see :ref:`plugin-params`).

Inversely, saved parameters in a plugin specific xml-file can also be set after that the plugin instance has been created and initialized (with the mandatory and optional parameters given by the constructor in python). The loading of these xml-parameters is dependent on the value of the member variable **m_autoLoadPolicy** of the plugin's interface class. This variable can have the following values:

.. code-block:: c++

    enum tAutoLoadPolicy {
        autoLoadAlways           = 0x1, /*!< always loads xml file by addInManager */
        autoLoadNever            = 0x2, /*!< never automatically loads parameters from xml-file (default) */
        autoLoadKeywordDefined   = 0x4  /*!< only loads parameters if keyword autoLoadParams=1 exists in python-constructor */
    };

If the variable is set to *autoLoadAlways*, all parameters in the xml-subtree, having the same unique identifier (ID) than the recently opened plugin instance, are set in the instance. This is done by calling the method **setParam** of the instance for every specific parameter. It is up to you to decide whether you accept the given parameter from the xml-file or not. Remember that **setParam** is called after that the plugin instance is created and the init-method of the plugin has been called. Of course, the init-method gets the mandatory and optional parameters, which are indicated when you create the plugin instance for example in python.

If the variable is set to *autoLoadNever*, no parameters of the xml-file will be set to the plugin's instance.

If the variable is set to *autoLoadKeywordDefined*, the parameters from the xml-subtree are only set to the plugin's instance, if the plugin has been created in python with the last keyword-defined parameter **autoLoadParams=1**.

Example:

.. code-block:: python

    a = dataIO("DummyGrabber",param1,...,paramN,autoLoadParams=1)
