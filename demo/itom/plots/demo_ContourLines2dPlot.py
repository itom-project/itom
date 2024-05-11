"""Contourlines 2D plot
=======================

This demo shows how to display contour lines in an ``itom.plot2``.

The contour levels can be set via the ``contourLevels`` property.
The expected type is a ``dataObject`` of shape ``1 x N`` of type
``uint8``, ``int8``, ``uint16``, ``int16``, ``int32``,
``float32`` or ``float64``. The line width and color map can
be adjusted by setting the property ``contourLineWidth`` and
``contourColorMap`` respectively."""

import numpy as np
from itom import dataObject
from itom import plot
# sphinx_gallery_thumbnail_path = '11_demos/_static/_thumb/demoContourLines2DPlot.png'

vec = np.linspace(-500, 501, 1001)
x, y = np.meshgrid(vec, vec)
r = np.sqrt(x**2 + y**2)

[idx, handle] = plot(r)
levels = dataObject.randN([1, 3], "uint8")
levels[0, 0] = 50
levels[0, 1] = 75
levels[0, 2] = 250

handle["contourLevels"] = levels
handle["contourColorMap"] = "hotIron"
handle["contourLineWidth"] = 5

###############################################################################
# .. image:: ../../_static/demoContourLine2DPlot_1.png
#    :width: 100%
