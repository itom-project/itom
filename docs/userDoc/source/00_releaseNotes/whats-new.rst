.. include:: ../include/global.inc

Changelog
############

This changelog only contains the most important changes taken from the commit-messages of the git.

**Head Revision**

* help dock widget to show a searchable, online help for the script reference of several python packages, the itom module as well as a detailed overview of algorithms and widgets contained in plugins
* revised documentation and python docstrings
* optimization due to tool *cppCheck*
* method *apiGetParam* added to api-functions
* timeouts changeable by ini-file
* size, position... of native dock widgets and toolbars is saved and reloaded at restart
* further demos added
* property editor for plots added
* compilation without PointCloudLibrary possible (CMake setting)
* easier compilation for linux
* 2DQwtPlot enhanced within a code sprint to support geometric primitives that can be painted onto the plotted image. The parameters of the geometries can then be obtained via python and for instance be evaluated in the designer widget 'evaluateGeometries'. Demo script added for demonstrating this functionality (*uiMeasureToolMain demo*)
* many methods of dataObject now have int-parameters or return values instead of size_t -> better compatibility with respect to OpenCV
* In :py:class:`itom.uiItem` it is now possible to also assign a string (or an integer) to enumeration based properties of widgets.
* :py:meth:`itom.openScript` enhanced to also open the script where any object (class, method...) is defined in (if it exists)
* :py:class:`itom.dataObject` have the attributes 'ndim' and 'shape' for a better compliance to 'numpy.ndarray'
* color type 'rgba32' added for data objects. See also :py:class:`itom.rgba`. The color class in C++ is contained in color.h. 2dgraphicview plot also supports color cameras. OpenCVGrabber can grab in color as well. Unittest of data object adapted to this.
* better exception handling of any exception occurring in plugin algorithms.
* type 'ui.DOCKWIDGET' now possible for :py:class:`itom.ui` in order to dock any user defined window into the main window
* drag&drop from last-command dock widget to console or script window
* modified python debugger
* added :py:meth:`itom.compressData` and :py:meth:`itom.uncompressData` in :py:mod:`itom`
* normalize method for data objects added
* many bugfixes

**Version 1.0.14**

there is no continuous changelog for these version
