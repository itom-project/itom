.. include:: ../include/global.inc

Python-Module |itom|
********************

The main purpose of the embedded |python| interpreter in |itom| is to access the specific functionalities provided by |itom|. This is done by the |python|-module :py:mod:`itom`, that
is importable only in the embedded |python| interpreter in |itom|. This module includes interfaces for hardware and algorithm plugins of |itom| (see :ref:`here for more information <getStartHardware>`) as well as classes that wrap the
most important internal data structures of |itom|, like matrices (class :py:class:`~itom.dataObject`, :ref:`link to documentation <itomDataObject>`), point clouds (class :py:class:`~itom.pointCloud`) 
or polygon meshes (class :py:class:`~itom.polygonMesh`). Additionally the module provides functions to manipulate or extend the graphical user interface of |itom| as well as to 
create own dialogs or windows (provided by the class :py:class:`~itom.ui` and :py:class:`~itom.uiItem`).

The full script reference of the module :py:mod:`itom` can be found under :ref:`itom-Script-Reference`.

Import |pyItom|
==============================

Like any other python module, you need to import the module |pyItom| in your script. Usually, the import is done at the beginning of a script. In the following example, it
is shown how the method **filterHelp** of the module |pyItom| can be called with respect to different ways of import |pyItom|:

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

However, if you import another script file in your main script file and you want to access any methods of |pyItom| in this secondary script, 
then you need to import |pyItom| in this script using one of the methods shown above.

Content of |pyItom|
-------------------

* class :py:class:`~itom.dataObject`. This is the |itom| internal matrix class, compatible to :py:class:`numpy.array`, and also used by any connected grabber or camera. Every matrix is extended by fixed and user-defined tags and keywords. For an introduction to the data object, see :ref:`itomDataObject`, a full reference is available under :py:class:`~itom.dataObject`. The data object is compatible to any numpy array, however the tags and keywords will get lost.
* classes :py:class:`~itom.ui` and :py:class:`~itom.uiItem` are the main classes for creating user defined dialogs and windows in |itom| and show them using some lines of script code. For more information about their use, see :ref:`qtdesigner` or their definitions in the script reference :py:class:`~itom.ui` and :py:class:`~itom.uiItem`.
* classes :py:class:`~itom.pointCloud` and :py:class:`itom.polygonMesh` are wrapper for point clouds and polygon mesh structures that are given by the optional integrated Point Cloud Library.
* class :py:class:`~itom.dataIO` is the class in order to access any plugin instance of type **dataIO** (cameras, grabbers, AD-converter...). The full reference can be found under :py:class:`~itom.dataIO`. More information about cameras and other dataIO devices can be found under :ref:`getStartGrabber` and :ref:`getStartADDA`.
* class :py:class:`~itom.timer` is the class to create a timer object that continuously calls a piece of Python code or a method with an adjustable interval.
* **class actuator** is the class in order to access any plugin instance of type **actuator**, like motor stages... The full reference can be found under :py:class:`~itom.actuator`. More information about actuator devices can be found under :ref:`getStartActuator`.
* **other class free methods**. |pyItom| also directly contains a lot of methods, that makes features of |itom| accessible by a |python| script. By these methods you can
    * add or remove buttons or items to the |itom| menu and toolbar (see :ref:`toolbar-start`)
    * get help about plugins and their functionality
    * call any algorithm or filter, provided by a plugin of type **algo**
    * directly plot matrices like dataObjects.
    
See :ref:`the itom script reference <itom-Script-Reference>` for a full reference to all classes and methods provided by the module |pyItom|.



