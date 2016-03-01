.. include:: /include/global.inc

.. _getStartFilter:

Getting started with algorithm plugins a.k.a. filters
************************************************************

Introduction
===============
Algorithm plugins may contain often used filters, specialised algorithms that work faster if implemented in C++ rather than python, or even complex software including e.g. GUIs and measurement control along with evaluation algorithms.
Each plugin can contain a number of algorithms, which are named **filters** due to historical reasons.
Like for every plugin, information on the filter is gained by selecting **Info...** from the context menu of your **plugin** or the actual **filter**. Here, you will find the mandatory and optional parameters of the specific filter. 

You can also get information on filters using the **filterHelp()** command in python. Without argument, it gives a list of all available filters including a short info string. If you know a part of your filter's name, the list will only contain filters using this name. If you use the complete name of filter, detailed help like in the GUI approach will be displayed.

.. code-block:: python
    :linenos:
    
    filterHelp()            # list of all available filters
    filterHelp('mean')      # list shortened to filters containing 'mean'
    filterHelp('calcMeanZ') # detailed information on 'calcMeanZ'


Usage of filters
=======================
Once you know the name of the filter you want to use, you can call it by use of the **filter()** command. It takes the filtername as string and the mandatory and optional parameters as arguments. Paramaters may be provided in designated order in keyword based notation.

.. code-block:: python
    :linenos:
    
    filter("[filtername]", mandParams, optParams)

.. note::
    
    Most filters require some sort of destination dataObject. Make sure to provide suitable dimensions and datatype.