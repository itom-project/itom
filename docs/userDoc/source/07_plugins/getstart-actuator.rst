.. include:: /include/global.inc

.. moduleauthor:: T. Boettcher, J. Krauter
.. sectionauthor:: J. Krauter



.. _getStartActuator:

Getting started with actuators
*********************************

Introduction
====================
This tutorial gives a short introduction to the use of the class :py:class:`~itom.actuator`. Similar tutorials are available for :ref:`grabbers <getStartGrabber>` and :ref:`A/D converters <getStartADDA>`.

Initialisation and common properties of the :py:class:`~itom.actuator` are described below.

Functions
====================
For this tutorial, we will focus on the standard application of actuators: move or rotate some stage. The axis numbers of the actuator are define `0` for x, `1` for y and `2` for z. If the actuator have only one axis, the number is `0`.

.. note::
    
    As one of the major advatages of the plugin concept, all actuators shall behave in the same manner when given the same command. There may however be some special properties of some devices, wich cause slightly different behavior in very specialised cases. For further information read the plugin documentation of your device.
    
Initialisation of an actuator
----------------------------------
Before using an actuator, we have to :ref:`initialise <initHardware>` it.  

.. code-block:: python
    :linenos:
    
    myactuator = actuator("[your plugin name]") # e.g. "dummyMotor"

.. note::
    
    Some actuators may start a calibration run after initialization. Ensure that there are no obstacles somewhere.
    
Move actuator
--------------
Actuators can be moved by using the function :py:meth:`~itom.actuator.setPosRel()` for relative move steps and :py:meth:`~itom.actuator.setPosAbs()` for absolut move steps. Depending on your application one of both may be better to use. Following command moves the *axis* of your actuator to the absolute *absPos* position in global actuator coordinates. It may be useful to run the calibration before usage. 

.. code-block:: python
    :linenos:
    
    myactuator.setPosAbs(axis, absPos)
    
If you want to move only a relative step *relStep* from your current position you can do it by this command:

.. code-block:: python
    :linenos:
    
    myactuator.setPosRel(axis, relStep)
    
    
Move actuator and take pictures
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Most times an actuator is used to move your object you want to measure and take a picture by a camera. First you need a running instance of your :py:class:`~itom.actuator` and :py:class:`~itom.dataIO` grabber (:ref:`Getting started with grabbers <getStartGrabber>`). Define the parameters of the camera and actuator :ref:`parameters <hardwareParameters>` before usage. Define your :py:class:`~itom.dataObject` to save the captured data. Then you need to define the actuator trajectory as absolute or relative positions. 

This example shows how you can move your object and take a picture at different positions using relative position steps:

.. code-block:: python
    :linenos:
    
    numberPos = 10 #number of positions
    relStepSize = 0.1 # relative step size 100um 
    axis = 2 #move in z direction
    sizeX = mycamera.getParam("sizex")
    sizeY = mycamera.getParam("sizey")
    dObj = dataObject([numberPos, sizeY, sizeX]) #define dataObject with the size: numberPos x sizeY x sizeX. 
    for cnt in range(0, numberPos): #loop to get at each position a camera picture
        myactuator.setPosRel(axis, relStepSize)
        d = dataObject()
        mycamera.acquire()
        mycamera.getVal(d)
        dObj[cnt,:,:] = d
        
.. note::
    
    Depending on your application it may be better first to acquire the camera and then to move the actuator. 

This example shows how you can do the same procedure using absolute actuator positions:

.. code-block:: python
    :linenos:
    
    numberPos = 10 #number of positions
    relStepSize = 0.1 # relative step size 100um 
    axis = 2 #move in z direction
    currentPos = myactuator.getPos(axis)
    sizeX = mycamera.getParam("sizex")
    sizeY = mycamera.getParam("sizey")
    dObj = dataObject([numberPos, sizeY, sizeX]) #define dataObject with the size: numberPos x sizeY x sizeX. 
    for cnt in range(0, numberPos): #loop to get at each position a camera picture
        myactuator.setPosAbs(axis, currentPos + cnt * relStepSize)
        d = dataObject()
        mycamera.acquire()
        mycamera.getVal(d)
        dObj[cnt,:,:] = d

.. note::
    
    Some actuators may have only the option to move in absolute positions. Hence, here the first acquisition position is as the **currentPos** position, because the first loop has for the variable **cnt** the value `0`. 


Parameters
==========
Most actuator plugins let you control the device's settings through a set of parameters. Common parameters are **speed**, **accel** or **async**. Some are read only. Parameters are checked and set by **getParam()** and **setParam()** as seen :ref:`here <hardwareParameters>`

.. in the section :ref:`Usage of hardware plugins <hardwareParameters>` before.

.. note::
    
    If you don't know the name of the parameter you want to check, try **getParamListInfo()**.

    
Synchronized/ Asynchronized move
----------------------------------
As default the actuators move command waits until the actuator has arrived the target position. With the parameter **async** you can deactivate the option and the itom script will not wait until the end of the movement. The next moving command will wait until the previously target is reached.

.. code-block:: python
    :linenos:
    
    myactuator.setParam("async", 1)
    

Use actuator in your own GUI
================================
If you are developing your own GUI and want to use the current position of each actuator axis, you can assign the actuator to the design widget **MotorController**. 

.. code-block:: python
    :linenos:
    
    # be gui.myMotorController a designer widget of type MotorController
    gui.myMotorController["actuator"] = myactuator 
