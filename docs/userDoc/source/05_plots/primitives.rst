.. include:: ../../include/global.inc

.. _primitives:

**Primitives** - Marking and Measuring
==============================================

Drawing items into a QWT-Plot
----------------------------------------------



Reading from a QWT-Plot
----------------------------------------------


Signals and Slots
----------------------------------------------


Indexing for geometric elements
----------------------------------------------

The geometricPrimitives is a struct within the c-Stuctur of the programm used for exchanging the geometric elements from plots to other elements.
The structur can be used rowise as dataObject or float32-lists

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
    
    
