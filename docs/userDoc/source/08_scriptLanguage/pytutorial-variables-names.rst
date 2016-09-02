

.. include:: ../include/global.inc

Variables and names 
====================

.. moduleauthor:: PSchau
.. sectionauthor:: PSchau


Using variables
-------------------

Now you can output things with :py:func:`print` and you can do math. The next step is to learn about variables. In programming, a variable is nothing more than a name for something and helps making the code easier to read. In |Python| the equal sign (``=``) is used to assign a value to a variable.

.. code-block:: python
    :linenos:
        
    width = 10
    depth = 5
    height = 20
    
    area_size = width * depth
    volume = area_size * height
    
    # Variables can be output with print()
    print(area_size)
    
    # The output of strings and variables can be combined
    print("Volume:", volume)

::

    50
    Volume: 1000

A value can be assigned to several variables simultaneously as seen below. Additionally, you will learn how to make strings that have variables embedded in them. You embed variables inside a string by using specialized format sequences (in this case ``%d``) and then putting the variables at the end with a special syntax.

.. code-block:: python
    :linenos:
        
    width = depth = height = 10
    
    print("The edge lengths of the cube are %d, %d and %d, respectively." % (width, depth, height))
    print("Still the same volume: %d" % (width * depth * height))

::

    The edge lengths of the cube are 10, 10 and 10, respectively.
    Still the same volume: 1000

.. note:: Remember to put ``# -- coding: utf-8 --`` at the top of your script file if you use non-ASCII characters and get an encoding error.