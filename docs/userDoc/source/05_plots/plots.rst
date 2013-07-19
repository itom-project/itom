Plots and Figures
=========================

itom contains functionalities to plot the mainly supported data structures like dataObjects or compatible numpy arrays. The visualization of more complex
polygon mesh or point cloud structures is not part of the core application but can be provided by further plugins. Plots can be directly plot into their own
figure window, they can be nested as subplots into one common figure and it is also possible to integrate plots into user-defined widgets and windows
(see :ref:`qtdesigner`).

Such that the integration into user interfaces, created at runtime by the external Qt Designer application, is feasable, every different type of plot
is a plugin that implements the interface given by Qt Designer plugins. Therefore you can handle every itom plot as any other widget in Qt Designer, if
this tool is called by itom.

Usually itom comes with a set of default plots, that can be used for the following plotting tasks:

* Static plot of a 1D data object (line plot).
* Static plot of a 2D data object, where the matrix values can be colorized by different color maps (default: gray map). For data objects with higher dimensions it is usually possible to change the currently depicted plane.
* Live plot of a line camera device (line plot).
* Live viewer for a camera providing two dimensional data objects (image plot).
* Three-dimensional plotting of 2D data objects where the data is interpreted as 2.5D, such that each value is considered to be a height value (isometric plot).

See the following files in order to learn more about plots and figures in itom:

Content:

.. toctree::
   :maxdepth: 1
   
   plotOverview
   figureManagement
   linePlots
   imagePlots
   isometricPlot

