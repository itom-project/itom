.. include:: ../include/global.inc

.. _ref-shape:

shape
***********

The class **shape** represents 2D geometric elements with floating (or subpixel) precision. Any shape is given by a set of base points
and an optional transformation matrix. The following shapes are currently defined:

* Point
* Line (a direct connection between two points)
* Rectangle (an rectangle that is mainly defined by the top-left and bottom-right corner point)
* Square (similar to a rectangle, however the side lengths are equal. It is defined by the center point and the side lengths)
* Ellipse (an ellipse that is defined by the top-left and bottom-right corner of its outer rectangle)
* Circle (similar to an ellipse, however the side lengths are equal. It is defined by the center point and the side lengths)
* Polygon (polygon with n points)

Examples for the creation of shapes are:

.. code-block:: python
    
    point = shape(shape.Point, (0,0))
    line = shape(shape.Line, (0,0), (100,50))
    rect = shape(shape.Rectangle, (20,20), (70,100)) #top-left, bottom-right
    square = shape(shape.Square, (30,-50), 20) #center, side-length
    ellipse = shape(shape.Ellipse, (-50,-70), (-20, 0)) #top-left, bottom-right
    circle = shape(shape.Circle, (-30, 100), 40) #center, side-length

If the optional transformation matrix (2x3 float64 matrix) is set, the shape can be translated and/or rotated. Please consider, that any
rotation is currently not supported in any plot. Rectangles, squares, ellipses or circles are always defined, such that their main axes
are parallel to the x- and y-axis. Use the rotation to choose another principal orientation. The base points of the shape are never affected
by any transformation matrix. Only the contour points can be requested with the applied coordinate transformation (if desired).

It is possible to obtain a :py:class:`~itom.region` from any shape with a valid area (points and lines don't have an area). The region is always
a pixel-precise structure. Regions can be combined using union or intersection operators.

Furthermore, a mask :py:class:`~itom.dataObject` can be obtained from any dataObject using the method :py:meth:`~itom.dataObject.createMask` if one or multiple shapes are given.

The demo script *demoShapes.py* show further examples about the usage of shape objects.

.. currentmodule:: itom

.. autoclass:: itom.shape
    :show-inheritance:
    :members: