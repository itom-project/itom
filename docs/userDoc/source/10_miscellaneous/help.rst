.. include:: /include/global.inc

How to use the help
************************

In general you can use this help as any other help.

A problem with function references is, it is hard to keep up to date with the manual. In addition the help can never have all information about any possible
filter- or hardware-plugin delivered from a third party. Therefore some tools for online help were implemented into python and itom.

With help(method) you will get an online help for the method or module.
To get something similar for the plugin-system, the functions *filterHelp(...)*, *widgetHelp(...)* and *pluginHelp(...)* can be used.
If you already have a plugIn of type :py:class:`~itom.actuator` or :py:class:`~itom.dataIO` you can get an online-help for possible *parameters* via the member method *getParamListInfo()* and for the exec-functions use *getExecFuncInfo()*.

Rebuild the Help
==================
If you think your help is not up to date and you are using the itom development environment you can rebuilt your help.
Therefore you need the up-to-date-version of sphinx for python.
