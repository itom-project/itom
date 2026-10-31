.. include:: ../include/global.inc

.. _itom-methods:

itom methods
*************

This section shows the spezial itom methods.

.. currentmodule:: itom

Plotting and camera
======================
Camera plugins have the special method *liveImage*.

.. autofunction:: itom.liveImage

.. autofunction:: itom.plot

.. autofunction:: itom.plot1

.. autofunction:: itom.plot2

.. autofunction:: itom.plot25

.. autofunction:: itom.close

Methods for getting help and information about filters, plugins, ui-elements, ...
=================================================================================
You can get the help string in the itom command line for plugins, widgets, filters, ...

.. autofunction:: itom.pluginHelp

.. autofunction:: itom.widgetHelp

.. autofunction:: itom.filterHelp

.. autofunction:: itom.pluginLoaded

.. autofunction:: itom.version

Adding elements to the GUI
==========================
You can add buttons and menus to the itom GUI.

.. autodata:: itom.MENU

.. autodata:: itom.BUTTON

.. autodata:: itom.SEPARATOR

.. autofunction:: itom.addButton

.. autofunction:: itom.removeButton

.. autofunction:: itom.addMenu

.. autofunction:: itom.removeMenu

.. autofunction:: itom.dumpButtonsAndMenus

For more information about using these methods, see :ref:`toolbar-start`.

Save and load dataObjects to or from file
=========================================
Dataobject can be save as IDC files.

.. autofunction:: itom.loadDataObject

.. autofunction:: itom.saveDataObject

.. autofunction:: itom.loadMatlabMat

.. autofunction:: itom.saveMatlabMat

.. autofunction:: itom.loadIDC

.. autofunction:: itom.saveIDC

Debug-Tools
===========
These are some debug tools.

.. autofunction:: itom.gcEndTracking

.. autofunction:: itom.gcStartTracking

.. autofunction:: itom.getDebugger

Request user rights
===================
These are some methods to get user information.

.. autofunction:: itom.userIsAdmin

.. autofunction:: itom.userIsDeveloper

.. autofunction:: itom.userIsUser

.. autofunction:: itom.userGetInfo


Further commands
================
Some further commands.

.. autofunction:: itom.copyStringToClipboard

.. autofunction:: itom.scriptEditor

.. autofunction:: itom.openScript

.. autofunction:: itom.newScript

.. autofunction:: itom.setCurrentPath

.. autofunction:: itom.getAppPath

.. autofunction:: itom.getCurrentPath

.. autofunction:: itom.getScreenInfo

.. autofunction:: itom.checkSignals

.. autofunction:: itom.getDefaultScaleableUnits

.. autofunction:: itom.processEvents

.. autofunction:: itom.scaleValueAndUnit

.. autofunction:: itom.setApplicationCursor

.. autofunction:: itom.clc

.. autofunction:: itom.autoReloader

.. autofunction:: itom.getPalette

.. autofunction:: itom.setPalette

.. autofunction:: itom.getPaletteList

.. autofunction:: itom.showHelpViewer

.. autofunction:: itom.getQtToolPath

.. autofunction:: itom.registerResource

.. autofunction:: itom.unregisterResource

.. autofunction:: itom.clearAll
