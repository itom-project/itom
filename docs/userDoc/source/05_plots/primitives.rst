.. include:: ../../include/global.inc

.. _primitives:

**Primitives** - Marking and Measuring
==============================================

The plot-widgets itom1DQwtPlot and itom2DQwtPlot supports plotting of geometric primitives by user interaction and script language.
This section will give a short introduction about ploting, read- /write-functions and the correspondig plots and the internal geometric element structure.

At last the evaluateGeomtrics-plugin for direct evaluation of geometric elements is introduced.

Drawing items into a QWT-Plot
----------------------------------------------

.. figure:: images/drawInto2D.png
    :scale: 70%
    
.. figure:: images/drawInto1D.png
    :scale: 70%


Reading from a QWT-Plot
----------------------------------------------



Signals and Slots
----------------------------------------------



Indexing for Geometric Elements
----------------------------------------------

The geometricPrimitives is a struct within the c-Stuctur of the programm used for exchanging the geometric elements from plots to other elements.
The structur can be used rowise as dataObject or float32-lists

At the moment only tPoint, tLine, tEllipse and tRectangle are supported.

The cells contain:

1. The unique index of the current primitive, castable to int32 with a maximum up to 16bit index values

2. Type flag 0000FFFF and further flags e.g. read&write only FFFF0000

3. First coordinate with x value

4. First coordinate with y value

5. First coordinate with z value
    

All other values depends on the primitiv type and may change between each type.

* A point is defined as idx, flags, centerX0, centerY0, centerZ0
* A line is defined as idx, flags, x0, y0, z0, x1, y1, z1
* A ellipse is defined as idx, flags, centerX, centerY, centerZ, r1, r2
* A circle is defined as idx, flags, centerX, centerY, centerZ, r
* A rectangle is defined as idx, flags, x0, y0, z0, x1, y1, z1, alpha
* A square is defined as idx, flags, centerX, centerY, centerZ, a, alpha
* A polygon is defined as idx, flags, posX, posY, posZ, directionX, directionY, directionZ, idx, numIdx

.. toctree::
   :hidden:

.. doxygenclass:: ito::PrimitiveContainer
	:project: itom
	:members:
    
    
Evaluation of Geometric Elements 
----------------------------------------------

The evaluateGeomtrics-widget is designed to load geometric definition stored in a float32 dataObject with a column-size of >10 elments and a row for each geometric element to display.
Further more it allows the evaluation of geometric relations between the geometric primitives.
