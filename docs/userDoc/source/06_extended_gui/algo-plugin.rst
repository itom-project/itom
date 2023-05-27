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

The following part of the documentation mainly describes how to show the widget and interact with it using Python.

A list of available widgets can be obtained by the following command:

.. code-block:: python

    widgetHelp()
    #this command prints a list of all widgets into the command line

In order to create a desired widget, the initialization function might require a set of mandatory or optional parameters, like it is the
case for calling filters or instantiating :py:class:`~itom.dataIO` or :py:class:`~itom.actuator` instances. If you want to get the list of parameters,
call :py:func:`itom.widgetHelp` with the name of the widget as parameter (in difference to the argument-less call above). You will then see a printed
list in the command line (comparable to :py:func:`itom.filterHelp`).

If the widget does not require any parameters, it is possible to simply open it via the GUI of itom. In the plugin toolbox, call **Open Widget...** from the
context menu of the widget (child of the algorithm plugin entry where the widget is programmed).

In order to get an instance of the widget via Python (with or without mandatory or optional parameters), use the class-methods :py:meth:`~itom.ui.createNewPluginWidget`
or :py:meth:`~itom.ui.createNewPluginWidget2`. The first provides an easy access to the widget initialization, the latter let you parameterize the obtained
widget, similar to the constructor of the class :py:class:`~itom.ui`. In both cases, the methods return an instance of :py:class:`itom.ui` that can be used
like the user-defined user interfaces, described in section :ref:`qtdesigner`.

An example for the easy widget generation is as follows:

.. code-block:: python

    #we assume that a widget (created as main window) with the name 'my_window' is available.
    #it has one mandatory integer parameter (name: index)

    #the window is then created with deleteOnClose = False, childOfMainWindow = True and
    #a window type that is derived from the class of the widget (here: ui.TYPEWINDOW).
    easy_widget = ui.createNewPluginWidget("my_window", index = 5)
    easy_widget.show()

The default parameterizations have the same meaning than the parameters of the constructor of the class :py:class:`itom.ui`. If you want to
further changes these parameters, use the following form:

.. code-block:: python

    window = ui.createNewPluginWidget2("my_window", (), {"index":5}, type = ui.TYPEDOCKWIDGET, dockWidgetArea = ui.BOTTOMDOCKWIDGETAREA)
    window.show()

In the case of a GUI based initialization, the widget is always opened with the *deleteOnClose* flag set to True, such that the widget is destroyed
if the user closes it. In the script base approach, this value is False per default. It is closed and destroyed if the variable referencing to the corresponding
:py:class:`itom.ui` instance is deleted.
