.. include:: /include/global.inc

.. moduleauthor:: T. Boettcher, J. Krauter
.. sectionauthor:: J. Krauter

.. _getStartADDA:

Getting started with A/D converters
###################################

Introduction
*************
This tutorial gives a short introduction to the use of A/D converters, which are adressed by the :py:class:`~itom.dataIO` class. Similar tutorials are available for :ref:`actuators <getStartActuator>` and :ref:`grabbers <getStartGrabber>`.

Initialisation and common properties of :py:class:`~itom.dataIO` and :py:class:`~itom.actuator` are described :ref:`here <initHardware>`.

Functions
**********
For this tutorial, we will focus on the standard application of A/D converters: detecting or applying analog signals 

.. note::
    
    As one of the major advatages of the plugin concept, all A/D converters shall behave in the same manner when given the same command. There may however be some special properties of some devices, wich cause slightly different behavior in very specialised cases. For further information read the plugin documentation of your device.

Start/Stop device
=================
Before using a A/D converter, we have to start it. For easy use, most plugins start the device at initialisation. We recommend manual starting to be sure. After usage, the device may be stopped. This is done with the functions **startDevice()** and **stopDevice()**:

.. code-block:: python
    :linenos:
    
    myadda = dataIO("[Your plugin name]") #  i.e. "MeasurementComputing"
    myadda.startDevice()
    myadda.stopDevice()
    
Detecting analog signals
=========================
Most times a A/D converter is used to detect analog signal like the voltage of a photodiode receiver. The procedure is fairly easy: once you have a running instance of the converter, calling **acquire()** triggers the acquisition of the specified converter input pins. The data are then retrieved either by the **getVal()** or the **copyVal()** command. Most of the A/D converters allows the parallel acquisition of several input ports. Here, **getVal()** makes a shallow copy, and **copyVal()** gives you a deep copy. Both methods take a dataObject as argument, which has to have suitable size and data type. 

Simple example for the plugin MeasurementComputing. Here the input is defined between the input port channel 0 and 3. The dataObject will be of the size 4:

.. code-block:: python
    :linenos:
    
    dObj = dataObject() # no need for fixed size or data type here, we be defined by getVal command
    myadda = dataIO("MeasurementComputing", board_number) 
    high_channel = 3
    low_channel = 0
    myadda.setParam("analog_high_input_channel", high_channel)
    myadda.setParam("analog_low_input_channel", low_channel)
    myadda.acquire()
    myadda.getVal(dObj)
    
Setting analog output values
============================
A/D converters can also be used to apply analog output voltage to the specified output ports. This example shows with the MeasurementComputing adda plugin how to do so. Depending on the plugin and devices. This may be possible by using voltage or digital values:

.. code-block:: python
    :linenos:

    # set the analog output by digital values
    numberChannels = 2
    numberSamples = 1 
    outValues = dataObject([numberChannels, numberSamples], 'int16') 
    outValues[:,:] = 1023 # 5V analog output in case of 10bit output channel resolution
    myadda.setVal(outValues)

Parameters
==========
Most adda plugins let you control the device's settings through a set of parameters. Common parameters are Parameters are checked and set by **getParam()** and **setParam()** as seen in the section :ref:`Usage of hardware plugins <hardwareParameters>` before.

.. note::
    
    If you don't know the name of the parameter you want to check, try **getParamListInfo()**.    