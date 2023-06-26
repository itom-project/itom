.. include:: ../include/global.inc

Plugin documentation
==========================

Besides the user documentation, there is also a plugin documentation for all currently available plugins. On a setup installation, the main files
for the plugin documentation are contained in the subfolders *doc* of each plugin in the plugin directory of |itom|. Based on these files, |itom| scans
their modification date when the |itom| internal help is called for the first time. If the help needs to be rebuild, a bundle is collected from all plugin
sub-documentations and saved in the *docs/pluginDoc/build* directory of the build directory of |itom|. Finally this bundle is packed and prepared for the
help assistant of |itom|.

Create the plugin documentation of any plugin
----------------------------------------------

In order to generate the plugin documentation of any plugin, the following requirements need to be fulfilled:

1. In the sources of the plugin there must be a folder **docs** that contains at least on \*.rst with the plugin documentation. This documentation needs to
    be written in the so called reStructured-Text format rst (see http://sphinx-doc.org/rest.html)

2. The file **CMakeLists.txt** of the specific plugin must contain the following line in order to register the rst-file as plugin documentation file::

    itom_configure_plugin_documentation(${target_name} <filenameOfTheRstFile>) #the filename must not contain the suffix .rst.

3. If the plugin is build, its build folder will get a **docs** subfolder, too. This subfolder consists of a file **plugin_doc_config.cfg**.

If these requirements are given, start |itom| and execute the script **create_plugin_doc.py** in the **docs/pluginDoc** directory of the build directory of |itom|.
Then select the \*.cfg-file describing the plugin documentation in its specific build folder.

In order to simultaneously create the documentations of many plugins, execute **create_all_plugin_docs.py** and indicate the build folder that contains the build-subfolders
of many plugins. These subfolders are searched for appropriate \*.cfg files and all sub-documentations are created.

.. note::

    **How does this work under the hood?**

    The most important file for generating the necessary html and QtHelp files from the given rst-files, that are located in the source folder of a plugin, is the configuration file
    with suffix *cfg* that is located in the build folder of the specific plugin.

    This file is generated when CMake executes the CMakeLists.txt file of the plugin and if this file contains the macro **PLUGIN_DOCUMENTATION** like stated above. This macro
    generates the configuration file based on the template **plugin_doc_config.cfg.in** located in **itom/SDK/docs/pluginDoc**.

    The configuration file then consists of the name of the plugin, the located where the source rst-file is located, the plugin build directory where the intermediate files of each
    single plugin documentation are generated and finally the installation path of each plugin (subfolder plugin of the itom build directory) where important components of the plugin
    documentation will be copied after having been build.

    When executing the script **create_plugin_doc.py** like described above, the plugin documentation (rst-file) is compiled and build in the QtHelp format in the build directory of the plugin. Afterwards the important components are copied into the documentation installation path of the plugin, given by its configuration file.

    For creating the final file **itomPluginDoc.qch** in the help subfolder of itom, itom reads all docs subfolder of plugins lying in the plugin subfolder of itom. Then all plugin documentations are collected in the itom subfolder **docs/pluginDoc/build**. From all plugins, the startsite **index.html** is created and the file list, search indices, keywords... from the single plugin qhp-files are merged into the file **itomPluginDoc.qhp**. Finally the Qt Help is created using the qhelpgenerator and qcollectiongenerator tool and copied into the help subfolder of itom.


plugin documentation source files
----------------------------------

The source files of each plugin documentation are written in the reStructuredText-format (*.rst). You can use all possibilities given by this format including the additions
provided by **sphinx** (see http://sphinx-doc.org/rest.html). Additionally, when the documentation is build using |itom|, specific directives, roles and a pyitom-domain is added in
order to automatically create parts of the documentation depending on information that can for instance be obtained using the command :py:meth:`itom.pluginHelp`.

Use the following roles as placeholders in the text. The placeholders will be replaced with information obtained by the plugin with the given name. This is only possible if
this specific plugin is loaded in |itom|:

.. code-block:: rst

    :pluginsummary:`pluginname` -> short description of the plugin
    :plugintype:`pluginname` -> type of the plugin (DataIO, Actuator, Algorithm)
    :pluginlicense:`pluginname` -> license string
    :pluginauthor:`pluginname` -> author(s) of the plugin
    :pluginversion:`pluginname` -> current version string

Furthermore, there are directives that you can use in order insert more information into your documentation file:

For inserting the detailed description of the plugin, write:

.. code-block:: rst

    .. pluginsummaryextended::
        :plugin: pluginname

A table with all mandatory and optional parameters that are required to start an instance of a *dataIO* or *actuator* plugin, write:

.. code-block:: rst

    .. plugininitparams::
        :plugin: pluginname

The last directive is created for algorithm plugins. An list of all available filters is obtained via

.. code-block:: rst

    .. pluginfilterlist::
        :plugin: pluginname
        :overviewonly:

Omit the option *:overviewonly:* in order to get an extended overview of all filters including their mandatory, optional and return arguments.
If this overview is inserted all filters in the short list will link to their specific long description.
