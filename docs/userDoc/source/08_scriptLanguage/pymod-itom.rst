.. include:: ../include/global.inc

Python-Module |itom|
********************

The module |pyItom| is a python module, that builds the connecting part between the embedded python script language and the overall |itom| application.
This module is only available in the context of a running |itom| software.

Import |pyItom|
---------------

Like any other python module, you need to import the module |pyItom| in your script. Usually, the import is done at the beginning of a script. In the following example, it
is shown how the method **filterHelp** of the module |pyItom| can be called with respect to
different ways of import |pyItom|:

.. code-block:: python
    :linenos:
    
    #1. import the whole module. Any method is then directly accessible
    from itom import *
    filterHelp("...")
    
    #2. import the itom module without the global access to its methods
    import itom
    itom.filterHelp("...")
    
    #3. only import certain methods or classes of the module itom
    from itom import filterHelp as anyAlias
    anyAlias("...")

If you simply type any commands in the |itom| command line or if you directly execute
any script at the top level, you don't need import the module |pyItom|, since this module
already has been imported at startup of |itom| with

.. code-block:: python
    
    from itom import *

However, if you import another script file in your main script file and you want to access any methods of |pyItom| in this secondary script, then you need to import |pyItom| in this script using one of the methods shown above.

Content of |pyItom|
-------------------

* class **dataObject**. This is the |itom| internal matrix class, compatible to *Numpy*, and also used by any connected grabber or camera. Every matrix is extended by fixed and user-defined tags and keywords. For an introduction to the data object, see :ref:`itomDataObject`, a full reference is available under :py:class:`~itom.dataObject`. The data object is compatible to any numpy array, however the tags and keywords will get lost.
* classes **ui** and **uiItem** are the main classes for creating user defined dialogs and windows in |itom| and show them using some lines of script code. For more information about their use, see :ref:`qtdesigner` or their definitions in the script reference :py:class:`~itom.ui` and :py:class:`~itom.uiItem`.
* class **dataIO** is the class in order to access any plugin instance of type **dataIO** (cameras, grabbers, AD-converter...). The full reference can be found under :py:class:`~itom.dataIO`.
* **class actuator** is the class in order to access any plugin instance of type **actuator**, like motor stages... The full reference can be found under :py:class:`~itom.actuator`.
* **other class free methods**. |pyItom| also directly contains a lot of methods, that makes features of |itom| accessible by a |python| script. By these methods you can
    * add or remove buttons or items to the |itom| menu and toolbar (see :ref:`toolbar-start`)
    * get help about plugins and their functionality
    * call any algorithm or filter, provided by a plugin of type **algo**
    * directly plot matrices like dataObjects.


Contents:

.. toctree::
   :maxdepth: 1
   
   itom/dataObject.rst  
   itom/pluginsPython.rst


