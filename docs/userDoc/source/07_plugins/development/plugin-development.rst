.. include:: ../../include/global.inc

.. sectionauthor:: Marc Gronle

Development under C++
=======================

The functionality of *itom* can be enlarged by programming plugins (sometimes also called add-ins). Such a plugin is written in C++ and can be integrated into *itom*
using the possibilities which are given by *Qt* (plugin management of *Qt*). Additionally each plugin must fit to one of three possible interface classes, such that *itom*
is able to communicate with this plugin. Finally, the resulting plugin is compiled as a *dll*-file (on Windows) or as an appropriate *a*-file on Linux-based machines and must
be located in the *plugin*-folder of *itom* or any subfolder. Then, the plugin is automatically recognized at startup of *itom*.

Content:

.. toctree::
   :maxdepth: 1

   projectSettings.rst
   pluginStructure.rst
   plugin-externalLibraries.rst
   pluginInterfaceClass.rst
   pluginClass.rst
   plugin-dataIO.rst
   plugin-actuator.rst
   plugin-algo.rst
   pluginAutoLoadSavePolicy.rst
   plugin-params.rst
   plugin-paramsMeta.rst
   plugin-paramsValidate.rst
   plugin-sharedSemaphore.rst
   plugin-RetVal.rst
   plugin-itomAPI.rst
   plugin-dataObject.rst



