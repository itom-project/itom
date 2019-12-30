.. _PlotsAndFigures:

Plots and Figures
#################################################

itom contains functionalities to plot the mainly supported data structures like data objects, compatible numpy arrays, point clouds or polygon meshes. Plots can be opened as an individual main window, they can be nested as subplots into one common figure or it is also possible to integrate plots into user-defined widgets and windows (see :ref:`qtdesigner`).

In order to allow an integration of plots into user interfaces that are created by the external Qt Designer at runtime, every plot is appended by means of 
a plugin that implements the interface given by Qt Designer plugins. Therefore you can handle every itom plot as any other widget in Qt Designer, if
this tool is called by itom.

Usually itom comes with a set of default plots, that can be used for the following plotting tasks:

* Static plot of a 1D data object (line plot).
* Static plot of a 2D or higher dimensional data objects, where the matrix values can be colorized by different color maps (default: grey map). For data objects with higher dimensions it is usually possible to change the currently depicted plane.
* Live plot of a line camera device (line plot).
* Live viewer for a camera providing two dimensional data objects (image plot).
* Three-dimensional plotting of 2D data objects where the data is interpreted as 2.5D, such that each value is considered to be a height value (isometric plot).
* There exist further plot plugins for visualizing an arbitrary number of point clouds, polygonal meshes or other geometrical features like spheres, cylinders or boxes.

See the following files in order to learn more about plots and figures in itom:

Content:

.. toctree::
   :maxdepth: 1
   
   plotOverview
   figureManagement
   plotClasses
   linePlots
   imagePlots
   isometricPlot
   vtk3dplot
   markers
   shapes
   matplotlib
   primitives
   designerPlugins


There are further custom widgets for the QtDesigner which realized itom specific non-plotting functions. See section :ref:`listCustomDesignerWidgets` for the widget description.