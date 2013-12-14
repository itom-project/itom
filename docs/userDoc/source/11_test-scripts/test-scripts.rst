.. include:: ../include/global.inc

.. _demoScripts:

Demo scripts
##############

There are several python demo scripts available which demonstrate the use of |itom|. All these files are in the directory **demo**
The following list gives a short description of each demo.

|itom| Basics
**************

- **demoDataObject.py**

  | *Description*: Here you can learn the basic function of the dataObject.
  | *Keywords*: creating a dataObject; plot; shallow copy; deep copy; meta data
  
- **demoToolBar.py**

  | *Description*: Creating your own toolbar and buttons.
  | *Keywords*: create a new class; add new functions; add button


Plugins
*********

- **demoDummyGrabber.py**

  | *Description*: Usage of a camera plugin.
  | *Keywords*: dataIO; start device (camera); snapshot (getVal); live image

- **demoDummyMotor.py**

  | *Description*: Usage of a motor plugin.
  | *Keywords*: set position; get position
  
- **demoCMU1394.py**

  | *Description*: Firewire grabber for different cameras. PointGray Firefly, Sony SX 900, Sony XCD-X700...
  | *Keywords*: dataIO; start device (camera); Snapshot (getVal); Live image
  
  
Algorithm / Filter
********************

- **demoOpenCVFilter.py**
  
  | *Description*: Median filtering of a randomly filled image.

- **demoNumpy.py**
  
  | *Description*: Short demonstration of some linear algebra functions provided by Numpy (numeric package of python).

- **demoScipy.py**
  
  | *Description*: Using scipy and matplotlib to calculate the cross-correlation between two images. Scipy is a python package that contains more scientific algorithms.
  | *Keywords*: scipy, matplotlib

- **demoSignalSmooth.py**
  
  | *Description*: Further example on how to use matplotlib (plotting package of python) in itom.

ui
****

The subfolder "ui" contains some examples on how to create customized user interfaces in itom (see :ref:`qtdesigner`). E.g.:

- **uiMeasureToolMain.py**
  
  | *Description*: Advanced GUI which enables geometric plotting and measurements within a 2D-QWT-Plot. This file shows how to auto-connect to signals and how to use buttons. The corresponding ui-file is uiMeasureToolMain.ui.
