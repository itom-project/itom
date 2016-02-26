.. include:: /include/global.inc

.. moduleauthor:: T. Boettcher, J. Krauter
.. sectionauthor:: T. Boettcher

.. _getStartGrabber:

Getting started with grabbers
*********************************

Introduction
===============
This tutorial gives a short introduction to the use of cameras. Cameras, RawIO and A/D converters (adda) are subtypes of the class :py:class:`~itom.dataIO`. 

.. kann der Satz weg?
.. Other IO Harware not matching the command sets of grabbers is the class :py:class:`~itom.actuator`.

Similar tutorials are available for :ref:`actuators <getStartActuator>` and :ref:`A/D converters <getStartADDA>`.

Initialisation and common properties of :py:class:`~itom.dataIO` and :py:class:`~itom.actuator` are described :ref:`getting here <getStartHardware>`.

Functions
=================
For this tutorial, we will focus on the standard application of grabbers: observe stuff, take images and retrieve them. 

.. note::
    
    As one of the major advatages of the plugin concept, all grabbers shall behave in the same manner when given the same command. There may however be some special properties of some devices, which cause slightly different behavior in very specialised cases. For further information read the plugin documentation of your device.

Start/Stop device
-----------------------
Before using a grabber, we have to start it. For easy use, most plugins start the device at initialisation. We recommend manual starting to be sure. After usage, the device may be stopped. This is done with the functions **startDevice()** and **stopDevice()**:

.. code-block:: python
    :linenos:
    
    mygrabber = dataIO("dummyGrabber")
    mygrabber.startDevice()
    mygrabber.stopDevice()
    

Autograbbing
-----------------------
The autograbbing feature sets a grabber into free-run mode, which allows for live view. Depending on the plugin's programming, autograbbing may be either enabled or disabled on initialisation. Therefore, it is recommended to set it explicitly to be sure.
Status is checked and set by the functions **getAutoGrabbing()** and **setAutoGrabbing()**:

.. code-block:: python
    :linenos:
    
    mygrabber = dataIO("dummyGrabber")
    mygrabber.startDevice()  #should be already started
    mygrabber.getAutoGrabbing()
    mygrabber.setAutoGrabbing(1)
    

Enabling and disabling autograbbing is also possible through functions **enableAutoGrabbing()** and **disableAutoGrabbing()**

.. code-block:: python
    :linenos:
    
    mygrabber = dataIO("dummyGrabber")
    mygrabber.startDevice()  #should be already started
    mygrabber.enableAutoGrabbing()
    mygrabber.disableAutoGrabbing()
    
.. note::
    
    If you experience problems when taking pictures manually/scripted, ensure that autograbbing is off

    
LiveImage
-----------------------
A grabber can provide a live image using the **liveImage()** command (if autograbbing in on):

.. code-block:: python
    :linenos:
    
    mygrabber = dataIO("dummyGrabber")
    liveImage(mygrabber)
    
.. note::
    
    Calling **liveImage()** starts a stopped device, but does not enable autograbbing.
    

Taking pictures
-----------------------
Most times a grabber is used, one would like to acquire and store pictures. The procedure is fairly easy: once you have a running instance of your grabber, calling **acquire()** triggers an image to be acquired. The image is then retrieved either by the **getVal()** or the **copyVal()** command. Here, **getVal()** makes a shallow copy, and **copyVal()** gives you a deep copy. Both methods take a dataObject as argument, which has to have suitable size and data type.

Simple example:

.. code-block:: python
    :linenos:
    
    dObj = dataObject() # no need for fixed size or data type here, will be defined by getVal command
    mygrabber = dataIO("dummyGrabber") # use standard init values
    mygrabber.startDevice()
    mygrabber.setAutoGrabbing(0)
    mygrabber.acquire()
    mygrabber.getVal(dObj) # this is a shallow copy. Use mygrabber.copyVal(dObj) for a deep copy alternatively
    mygrabber.stopDevice()

Most measurement controls afford multiple acquisitions. In this case, the dataObject to contain the picture stack has to be initialised using correct size and data type.

Multi-shot example:

.. code-block:: python
    :linenos:
    
    dObj = dataObject([10,100,200], dtype = "uint8") 
    # Here we need fixed size and data type, because we will assign layer by layer. 
    # Note inversed dimensions compared to grabber init!
    mygrabber = dataIO("dummyGrabber", 200,100,8)
    mygrabber.startDevice()
    mygrabber.setAutoGrabbing(0)
    for cnt in range(0,10):
        mygrabber.acquire()
        mygrabber.copyVal(dObj[cnt,:,:]) 
    mygrabber.stopDevice()
    
.. note::
    
    If you experience problems when taking pictures manually/scripted, ensure that autograbbing is off


Parameters
=================
Most grabber plugins let you control the device's settings through a set of parameters. Common parameters are **integration_time**, **roi**, **bpp**, or **binning**. Some are read only. Parameters are checked and set by **getParam()** and **setParam()** as seen in the section :ref:`Usage of hardware plugins <hardwareParameters>` before.

.. note::
    
    If you don't know the name of the parameter you want to check, try **getParamListInfo()**.


integration_time
-----------------------
Integration time is probably the most often changed parameter of a grabber. And also the simplest. This parameter changes the time over which each triggered frame is integrated.

.. code-block:: python
    :linenos:
    
    mygrabber.setParam("integration_time", 0.01) # integration time is set to 10ms

roi
-----------------------
Many grabbers support changing of the region of interest (ROI). This property is set by the parameter **roi** which contains four values: x0, y0, width, height. All four values have to be defined:

.. code-block:: python
    :linenos:
    
    mygrabber = dataIO("dummyGrabber")
    mygrabber.setParam("roi", [12,8,100,200])

.. note::
    
    Though it is unlikely, you may possibly come across older plugin versions that use an outdated notation, where the ROI is set via parameters **x0**, **x1**, **y0**, **y1** defining starting end ending pixels rather than starting pixel and length.


bpp
-----------------------
If a grabber provides control over the bitdepth of the output, it is set by the parameter **bpp** (bits per pixel). 

.. code-block:: python
    :linenos:
    
    mygrabber.setParam("bpp", 10) # bpp is set to 10


gain
-----------------------
This parameter gives access to a gain factor, if it is provided by the grabber. This factor may be arbitrarily changed, or be a binary value (for example for IR-mode on PCO PixelFlys).

.. code-block:: python
    :linenos:
    
    mygrabber.setParam("gain", 1) # gain is set to 100%

binning
-----------------------
For means of noise reduction and/or speed-up, some grabbers support binning of pixels.  

.. code-block:: python
    :linenos:
    
    mygrabber.setParam("binning", 202) # binning is set to 2x2
        
    
Use grabbers in your own GUI
================================
If you are developing your own GUI and want wo use live images from a grabber, you can assign the grabber as a source for your designer widget, just like you can display dataObjects.

.. code-block:: python
    :linenos:
    
    # be gui.myplot a designer widget of type Itom2dQwtPlot or Itom1dQwtPlot
    gui.myplot["camera"] = mygrabber # liveImage of grabber, if autograbbing in enabled
    gui.myplot["source"] = mydataobject # diplaying dataObject

Demo script cameraWindow.py
================================
A demo script named **cameraWindow.py** is provided, which demonstrates basic grabber use. Detailed discussion of this script is found below.


.. toctree::
    :maxdepth: 1
    
    demoCameraWindow.rst
    