"""Pick points and markers
==========================

This demo shows how you can pick points and markers in the ``itom`` plot."""

from itom import dataObject
from itom import plot2
from itom import plotItem
# sphinx_gallery_thumbnail_path = '11_demos/_static/_thumb/demoPickPointsAndMarkers.png'

###############################################################################
# **Pick Points demo**
#
# Create a random 2 dimensional ``dataObject`` and plot it.
obj = dataObject.randN([1024, 1024], "int16")

[nr, h] = plot2(obj)
h["title"] = "Showcase: pick marker"

###############################################################################
# .. image:: ../../_static/demoPickPointsAndMarkers_1.png
#    :width: 100%

###############################################################################
# This command let the user pick maximum 4 points (earlier break with space, esc aborts the selection).
pickedPoints = dataObject()
h.pickPoints(pickedPoints, 4)

print("coordinates of selected points: ")
for numPoint in range(pickedPoints.shape[1]):
    print(f"x: {pickedPoints[0, numPoint]}, y: {pickedPoints[1, numPoint]}")

###############################################################################
# Plot the ``dataObject`` again together with the previously selected points as ``marker``.
#
# The second argument of ``plotMarkers`` is a style-string (this may change) ``[color, symbol, size]``:
#
# +----------+------------------------------------------+
# | color    | \{b, g, r, c, m, y, k, w\}               |
# +----------+------------------------------------------+
# | symbol   | \{., o, s, d, >, v, ^, <, x, `*`, +, h\} |
# +----------+------------------------------------------+
# | size     | any integer number                       |
# +----------+------------------------------------------+
#
markers = dataObject([2, 3], "float32", data=[10.1, 20.2, 30.3, 7, 100, 500])
[nr, h] = plot2(obj)
h["title"] = "Showcase: plot the currently selected points"
h.call(
    "plotMarkers", pickedPoints, "b+10", "setName"
)  # 'setName' is the name for this set of markers (optional)

###############################################################################
# .. image:: ../../_static/demoPickPointsAndMarkers_2.png
#    :width: 100%

###############################################################################
# Delete marker set
h.call("deleteMarkers", "setName")  # deletes given set
h.call("deleteMarkers", "")  # deletes all sets

###############################################################################
# **Paint geometric shapes**
#
# Create a random 2 dimensional ``dataObject`` and plot it.
obj = dataObject.randN([1024, 1024], "int16")
[nr, h] = plot2(obj)
h["title"] = "Showcase: paint 4 ellipses"

###############################################################################
# This command let the user pick maximum 4 points (earlier break with space, esc aborts the selection).
geometricShapes = h.drawAndPickElements(plotItem.PrimitiveEllipse, 4)

print("selected shapes:")
for shape in geometricShapes:
    print(shape)

###############################################################################
# Plot the ``dataObject`` again together with the previously painted ellipses ``geometricShapes``.
[nr, hDrawInto] = plot2(obj)
hDrawInto["title"] = "Showcase: plot painted ellipses"
hDrawInto.call(
    "setGeometricShapes", geometricShapes
)  # "b" and "setname" will be ignored anyway
shapes = hDrawInto["geometricShapes"]

###############################################################################
# .. image:: ../../_static/demoPickPointsAndMarkers_3.png
#    :width: 100%
