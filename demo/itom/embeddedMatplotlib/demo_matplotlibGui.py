"""Matplotlib embedded in GUI
=============================

This examples shows how the ``matplotlib`` can be integrated
into a ``GUI`` based on the ``MatplotlibPlot`` Qt Designer plugin. 
"""
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from itom import ui

matplotlib.use("module://mpl_itom.backend_itomagg")
# sphinx_gallery_thumbnail_path = '11_demos/_static/_thumb/demoMatplotlibGUI.png'

###############################################################################
# Plots spines into the ``MatplotlibPlot`` Qt Designer plugin.
def plotDroppedSpines():
    """
    plot taken from matplotlib example 'spines_demo_dropped.py'
    """
    canvas = gui.plot  # reference to matplotlibPlot widget
    fig = plt.figure(num=3, canvas=canvas)
    ax = fig.add_subplot(111)
    ax.clear()

    image = np.random.uniform(size=(10, 10))
    ax.imshow(image, cmap=plt.cm.gray, interpolation="nearest")
    ax.set_title("dropped spines")

    # Move left and bottom spines outward by 10 points
    ax.spines["left"].set_position(("outward", 10))
    ax.spines["bottom"].set_position(("outward", 10))
    # Hide the right and top spines
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    # Only show ticks on the left and bottom spines
    ax.yaxis.set_ticks_position("left")
    ax.xaxis.set_ticks_position("bottom")

    plt.show()


###############################################################################
# .. image:: ../../_static/demoMatplotlibGUI_1.png
#    :width: 75%

###############################################################################
# Plots a sine curve into the ``MatplotlibPlot`` Qt Designer plugin.
def plotSine():
    """
    plots sine, taken from matplotlib gallery examples
    """
    t = np.arange(0.0, 1.0, 0.01)
    s = np.sin(2 * np.pi * t)

    canvas = gui.plot  # reference to matplotlibPlot widget
    fig = plt.figure(num=3, canvas=canvas)
    ax = fig.add_subplot(111)
    ax.clear()
    ax.plot(t, s)

    plt.show()


gui = ui("matplotlibGui.ui", type=ui.TYPEWINDOW)
gui.btnSine.connect("clicked()", plotSine)
gui.btnDroppedSpines.connect("clicked()", plotDroppedSpines)
gui.show()

# if you call this script for the second time, the given figure-num (3)
# is already in used for the lastly closed figure. Therefore also tell
# matplotlib to close this figure handle.
plt.close(3)

###############################################################################
# .. image:: ../../_static/demoMatplotlibGUI_2.png
#    :width: 75%
