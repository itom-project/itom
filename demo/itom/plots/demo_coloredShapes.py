# coding=utf8

"""Colored shapes
=================

This demo script shows how to set the line color of geometric shapes.

Per default, the color of geometric shapes depend on the currently applied
color map. Every color map has two additional inverse colors which should provide
a high contrast to other colors that appear within the color map.

Usually, the line (and optional filling) of a geometric shape is drawn using the inverse color #1.
If it is allowed to change the size and / or position of a shape or to rotate the shape,
the pickable squares of a selected shape is drawn with the inverse color #2.

While the behaviour of the latter (pickable squares) cannot be changed, the line color
of shapes can also be set permanently for each shape. This is done by the 'color' property
of the itom.shape object. If the color property is set to None, the automatic line color mode is selected,
else an object of type itom.rgba can be assigned.
"""
from itom import dataObject
from itom import rgba
from itom import shape
from itom import plot
# sphinx_gallery_thumbnail_path = '11_demos/_static/_thumb/demoColorShapes.png'


image = dataObject.zeros([200, 400], "uint8")
image[0:100, 0:200] = 128
image[100:, 200:] = 255
# coordinates of shapes are always given in 'physical' scales
# (considering scale and offset of the displayed object)
image.axisScales = (0.25, 0.25)
image.axisOffsets = (100, 200)

rect1 = shape.createRectangle((10, 5), (40, 20))
# change the color of the upper right rectangle to a permanent color (dark green)
rect1.color = rgba(0, 128, 55)
rect2 = shape.createRectangle(center=(0, 0), size=(40, 20))
rect2.color = None  # automatic, color map dependent coloring (default)

circle = shape.createCircle(center=(-25, -12), radius=5)
circle.color = rgba(120, 0, 90)  # change the color of the circle to a permanent color (purple)

ellipse = shape.createEllipse(center=(-20, 0), size=(40, 5))
ellipse.color = None  # automatic, color map dependent coloring (default)

[idx, handle] = plot(image, properties={"title": "Color shapes demo"})
handle["geometricShapes"] = (rect1, rect2, circle, ellipse)

###############################################################################
# .. image:: ../../_static/demoColoredShapes_1.png
#    :width: 100%
