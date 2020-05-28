.. include:: ../include/global.inc

.. |mm| unicode:: U+00B5 m

Changelog Version 4.0.0
#########################

itom
********

**Version 4.0.0 (2020-05-28)**

(more than xx commits in itom repository)


Plugins
******************

**Version 4.0.0 (2020-05-28)**

(xx commits in plugins repository)




Designer Plugins
******************

**Version 4.0.0 (2020-05-28)**

(more than 58 commits in designerPlugins repository)

* general: CMakeLists.txt and *.cmake files adapted to (new) cmake style guide of itom (see documentation/13_contributing/cmake_style_guide.rst). General settings of itom libraries as well as plugin and designer plugin libraries are now added in the INIT_ITOM_LIBRARY() macro of ItomBuildMacros.cmake. Include ItomBuildMacros.cmake at the beginning of the file and call INIT_ITOM_LIBRARY, such that also CMake policies are globally set. ITOM_SDK_DIR is now "auto"-detected in the overall CMakeLists.txt file.
* general: Remove Qt4 support and compiler switches within CMake scripts and source code
* general: Merged in algorithmInterrupt (pull request #8)
* general: plugin version numbers of plot designer plugins incremented
* general: adaptions of plot designer plugins with respect to modified base classes AbstractNode, AbstractFigure and Channel.
* general: for the versioning of the interface to itom designer plugins, it is necessary to update and build itom and change the Q_PLUGIN_METADATA macro in the major header file of each itom designer plugin to " Q_PLUGIN_METADATA(IID "org.qt-project.Qt.QDesignerCustomWidgetInterface" FILE "pluginMetaData.json")"
* general: fixes and improvements due to analysis of deepcode.ai
* general: Changed nullptr to NULL to stay compatible with older compilers

* itom2dqwtplot: grid property added (like in itom1dqwtplot)
* itom2dqwtplot: fix_issue#32 Linux linker Issue
* itom2dqwtplot: fixes #26. Changing the display mode led to the loss of x-data. refreshPlot was called without xData argument.
* itom2dqwtplot: fixes #25. Plane offset wasn't taken into account
* itom2dqwtplot: fixes issue #29 (please update itom before): a lot of refactoring in itom2dQwtplot; new property volumeCutPlotItem added to assign a given plot to show possible volume cuts. All properties lineCutPlotItem, zSlicePlotItem and volumeCutPlotItem can now be set to None in order to remove a previous connection to an existing plot.
* itom1dQwtPlot: fix for handling NaN values in line plot (if doAlign property is enabled). Be careful: qRound(NaN) returns 0 and not NaN (probably changed for newer Qt versions)
* itom1dQwtplot: if dataObject has an unit for the horizontal axis or the vertical value axis, this unit is added to the width/height of the distance between two pickers. For SI time or length units, widths or heights < 0.1 or >= 10000 are expressed in a more suitable unit, e.g. 20000 m -> 20 km. Shows no decimal values for integer distances (depending on dtype, axisScales, axisOffsets)
* itom1dQwtPlot: font of legend items has not always been assigned to the curves (e.g. if the font is selected in QtDesigner)
* itom1dQwtPlot: fixes issue #27: duplicated, adjacent points in X/Y data will be skipped when navigating through the points using the arrow keys
* itom1dQwtPlot: fixes #15. Implementation of a ROI check to set the legend entries considering a possible ROI.
* itom1dqwtplot: fixes issue #30. If an itom1dqwtplot is currently connected to the line cut / z-stack cut functionality of an itom2dqwtplot and the user assigns a new, different source to the source property of the line cut, the input parameters 'bounds' are resetted now, to fully display the newly given dataObject. However: the connection to the 2d plot is still there, hence, if the user draws a new line / z-stack position, this line is drawn again in the 1d plot. Avoid this by resetting the connection before assigning a new source object. This is done by assigning None to the properties 'lineCutPlotItem' / 'zSlicePlotItem' of the 2d plot.
* itom2dqwtplot, itom1dQwtPlot: removed unused #include of plotLegends (plotLegends do not exist any more in the SDK since they have been merged into itomWidgets a long time ago.
* itom2dqwtplot, itom1dQwtPlot: draw shapes on top of grid (z-index: 10), curves + images (z-index: 20) with z-index: 25. markers have a z-index of 30 and further labels 150.
* itom2dqwtplot, itom1dQwtPlot: renamed private member, incremented version numbers
* itom2dqwtplot, itom1dQwtPlot: allow panning with middle mouse button at any time. fixes #18
* itom2dqwtplot, itom1dQwtPlot: qwt is now compiled as shared library. All shared-files between itom1dqwtplot and itom2dqwtplot are now included in a new library itomqwtplotbase (shared library, too). itom1dqwtplot and itom2dqwtplot now link against this new library. Reason: The safe check for inheritances using Qt and qobject_cast (which is very often used in Qwt) can not properly work if multiple dlls link against the same statically linked base library. The same holds, if the same Qt class is compiled multiple times in different DLLs which are finally loaded in the same project. Therefore a clear structure is necessary, which is implemented now (linux has to be verified)
* itom2dqwtplot, itom1dQwtPlot: targets qwt and itomQwtPlotBase do now provide necessary include directories for other targets linking against these targets (interface include directories). This is especially necessary for the directory where the parsed ui-file is generated.

* itom2DGraphicView: removed deprecated plugin "itom2DGraphicView"
* itom2DGraphicView: bugfix in graphicViewPlot to secure some getter methods of properties of designer widgets, since the QtDesigner reads every property at startup and some of these properties are not in a properly initialized state within QtDesigner.

* matplotlibPlot: fixes issue #28 to pass the position of the enter event to the Matplotlib backend. The backend now has a backward compatibility mode and emits the old-style signal for enter/leaveEvent as well as new signals with additional position information for the enterEvent (Qt5 only). Update itom as well!

* itomIsoGLFigure: ubuntu c++11 bugfix in itomIsoGLFigure plugin

