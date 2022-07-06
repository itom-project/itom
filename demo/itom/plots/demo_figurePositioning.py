"""Figure positioning
=======================

Demo for getting/setting the size and position of a figure.
"""
from itom import figure
from itom import dataObject
# sphinx_gallery_thumbnail_path = '11_demos/_static/_thumb/demoFigurePositioning.png'


fig = figure()
fig.plot(dataObject.randN([100, 200]))

fig["geometry"]

###############################################################################
# Frame of figure window is the entire window including any title bar and window frame.
#
# properties: ``frameGeometry, x, y``
print("figure frame geometry (x,y,w,h):", fig["frameGeometry"])
print("figure position (x,y):", fig["pos"])
print("x, y:", fig["x"], fig["y"])

###############################################################################
# The real plot area of the figure is accessible by ``geometry``, ``size``, ``width``, ``height``.
print("figure geometry (x,y,w,h):", fig["geometry"])
print("figure size (w,h):", fig["size"])
print("figure width:", fig["width"])
print("figure height:", fig["height"])

###############################################################################
# In order to change the outer position use the property ``pos``.
fig["pos"] = (0, 0)

###############################################################################
# Size change: property ``size``.
fig["size"] = (500, 400)

###############################################################################
# In order to change the inner position and size use the property ``geometry``.
fig["geometry"] = (100, 200, 300, 200)
