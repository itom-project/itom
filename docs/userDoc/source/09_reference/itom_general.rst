itom methods
=============

.. currentmodule:: itom

Plotting and camera
---------------------
.. automodule:: itom
    :members: liveImage, plot

Using algorithms and filters
-----------------------------

.. automodule:: itom
    :members: filter

    
Methods for getting help and information about filters, plugins, ui-elements, ...
---------------------------------------------------------------------------------
.. automodule:: itom
    :members: pluginHelp, widgetHelp, filterHelp, pluginLoaded, version

Adding elements to the GUI
--------------------------
.. automodule:: itom
    :members: addButton, removeButton, addMenu, removeMenu

For more information about using these methods, see :ref:`toolbar-start`.

Disk-IO
----------
.. automodule:: itom
    :members: loadDataObject, saveDataObject, loadMatlabMat, saveMatlabMat, loadIDC, saveIDC
    
Debug-Tools
--------------

.. automodule:: itom
    :members: gcEndTracking, gcStartTracking, getDebugger

    
Further commands
----------------
.. automodule:: itom
    :members: scriptEditor, openScript, newScript, setCurrentPath, getAppPath, getCurrentPath, getScreenInfo, checkSignals, getDefaultScaleableUnits, processEvents, scaleValueAndUnit, clc
      

.. Defines
.. ----------------  
.. automodule:: itom
..    :members: DATA, FILE
    

Another possibility to add methods to this page is to use the auto-summary function.
Since, the default-role property in conf.py  is set to 'autolink' and the auto-summary module is included,
small pages will be automatically created for each method in the following list and a hyperlink to this site is created:

.. currentmodule:: itom

With method signature:

.. autosummary::
    :toctree: generated
    
    itom.widgetHelp
    itom.pluginHelp
    itom.filterHelp

Without method signature:
    
.. autosummary::
    :toctree: generated
    :nosignatures:
    
    itom.liveImage
    itom.plot