.. include:: /include/global.inc

.. moduleauthor:: T. Boettcher, J. Krauter
.. sectionauthor:: J. Krauter

.. _getStartADDA:

Getting started with A/D converters
**************************************

Introduction
====================
This tutorial gives a short introduction to the use of A/D converters, which are addressed by the :py:class:`~itom.dataIO` class. Similar tutorials are available for :ref:`actuators <getStartActuator>` and :ref:`grabbers <getStartGrabber>`.

Initialisation and common properties of :py:class:`~itom.dataIO` and :py:class:`~itom.actuator` are described :ref:`here <initHardware>`.

Functions
==========
For this tutorial, we will focus on the standard application of A/D converters: detecting or applying analog signals

.. note::

    As one of the major advatages of the plugin concept, all A/D converters shall behave in the same manner when given the same command. There may however be some special properties of some devices, which cause slightly different behavior in very specialised cases. For further information read the plugin documentation of your device.

Start/Stop device
-----------------------
Before using a A/D converter, we have to start it. Then, single or continuous data acquisition may follow. After usage, the device has to be stopped. This is done with the functions :py:meth:`~itom.dataIO.startDevice` and :py:meth:`~itom.dataIO.stopDevice`:
Like in the case of grabbers, **startDevice** or **stopDevice** can be called multiple times in a sequence. Only the first call of **startDevice** will set the device to a state that is ready to acquire data and the last
call of **stopDevice** will reset this state. This is implemented by an internal counter.

.. code-block:: python
    :linenos:

    myadda = dataIO("[Your plugin name]") #  e.g. "MeasurementComputing"
    myadda.startDevice()
    myadda.stopDevice()

Detecting analog signals
-----------------------------
Most times an A/D converter is used to detect analog signal like the voltage of a photodiode receiver. The procedure is fairly easy: once you have a running instance of the converter,
calling :py:meth:`~itom.dataIO.acquire` triggers the acquisition of the specified converter input pins. The data are then retrieved either by the :py:meth:`~itom.dataIO.getVal()` or
the :py:meth:`~itom.dataIO.copyVal` command. Most of the A/D converters allows the parallel acquisition of several input ports. Here, :py:meth:`~itom.dataIO.getVal` makes a shallow
copy, and :py:meth:`~itom.dataIO.copyVal` gives you a deep copy. In the case of the shallow copy, pass an arbitrary initialized dataObject to :py:meth:`~itom.dataIO.getVal`. This
object is then reconfigured to the right type and size and will contain a shallow copy of the recently acquired data. However, once the device acquires new data, it is possible
that the content of the dataObject will be changed, too, since it is only a shallow copy (however: this is a fast method to get the data). In the latter case of the method
:py:class:`~itom.dataIO.copyVal` you have to either pass an empty dataObject or a dataObject whose type and current region of interest fits to the expected output of the recent
acquisition. Then, you will obtain a deep copy of the acquired data array into the allocated dataObject or a newly allocated dataObject with the deeply copied data if it has been empty before.

Depending on the plugin, it is possible to simultaneously acquire data from multiple input pins (denoted bye channels) as well as a series of data from each pin (denoted by number of samples).
Each channel is represented by one row in the dataObject, whereas the number of samples are put in this line.

Simple example for the plugin MeasurementComputing. Here the input is defined between the input port channel 0 and 3. The dataObject will therefore have 4 rows:

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
----------------------------------------------
A/D converters can also be used to apply analog output voltage to the specified output ports. This example shows with the MeasurementComputing adda plugin how to do so.
Depending on the plugin and devices. This may be possible by using voltage or digital values:

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
