.. include:: ../include/global.inc

.. _itom-methods:

itom methods
*************

This section shows the spezial itom methods. 

.. currentmodule:: itom

Plotting and camera
======================
Camera plugins have the special method *liveImage*. 

.. automodule:: itom
    :members: liveImage, plot, plot1, plot2, plot25 ,close

Using algorithms and filters
============================
Algorithms and filters are called by the method *filter*. 
    
.. automodule:: itom
    :members: filter

Methods for getting help and information about filters, plugins, ui-elements, ...
=================================================================================
You can get the help string in the itom command line for plugins, widgets, filters, ...
    
.. automodule:: itom
    :members: pluginHelp, widgetHelp, filterHelp, pluginLoaded, version

Adding elements to the GUI
==========================
You can add buttons and menus to the itom GUI. 
    
.. automodule:: itom
    :members: MENU, BUTTON, SEPARATOR, addButton, removeButton, addMenu, removeMenu, dumpButtonsAndMenus

For more information about using these methods, see :ref:`toolbar-start`.

Save and load dataObjects to or from file
=========================================
Dataobject can be save as IDC files. 

.. automodule:: itom
    :members: loadDataObject, saveDataObject, loadMatlabMat, saveMatlabMat, loadIDC, saveIDC
    
Debug-Tools
===========
These are some debug tools. 

.. automodule:: itom
    :members: gcEndTracking, gcStartTracking, getDebugger
    
Request user rights
===================
These are some methods to get user informations. 

.. automodule:: itom
    :members: userIsAdmin, userIsDeveloper, userIsUser, userGetInfo

    
Further commands
================
Some further commands. 

.. automodule:: itom
    :members: scriptEditor, openScript, newScript, setCurrentPath, getAppPath, getCurrentPath, getScreenInfo, checkSignals, getDefaultScaleableUnits, processEvents, scaleValueAndUnit, setApplicationCursor, clc, autoReloader, getPalette, setPalette, getPaletteList, showHelpViewer, getQtToolPath, registerResource, unregisterResource, clearAll
