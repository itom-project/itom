.. include:: /include/global.inc

Implement a  more complex GUI in a plugin (C/C++)
***************************************************

It is already possible to implement many different types of user-defined interfaces by the QtDesigner combined with some Python scripting (see section :ref:`qtdesigner`).
However this approach is limited with respect to the usage of external libraries or widgets that are currently not supported or available as designer plugin in the
QtDesigner application.

Nevertheless, it is possible to implement arbitrary GUIs in an algorithm plugin (written in C++ and Qt). Then, an instance the Python class :py:class:`itom.ui`
can also be created from this widget and the standard interaction like connecting to signals or calling slots of this widget is possible.

One or multiple of those widgets can be implemented in one itom plugin of type **algorithm**. For more detailed information about the programming, read
the documentation about :ref:`the algo plugin class <plugin-class-algo>`.

The following part of the documentation mainly describes how to show the widget and interact with it using Python. TODO.