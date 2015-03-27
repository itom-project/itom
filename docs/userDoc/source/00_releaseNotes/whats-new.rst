.. include:: ../include/global.inc

Changelog
############

This changelog only contains the most important changes taken from the commit-messages of the git.

itom
********

**Version 1.4.0 (2015-02-17)**

(more than 200 commits in itom repository)

* improved getting started documentation
* CMake adapted for detection Visual Studio versions > 2010
* dockWidgetArea added as optional parameter for :py:class:`itom.ui`-class such that windows of type ui.TYPEDOCKWIDGET can be placed at different locations.
* find_package(OpenMP) added to ItomBuildMacros for all compilers, since Visual Studio Express does not provide this tool.
* improved drag&drop from variable names from workspace to console
* redesigned user management dialogs
* python access key to variables in workspace widget can now be inserted into console or scripts via drag&drop, detailed variables window also contain the right access key for dicts, list, tuples, attributes...
* itom.plotItem.PrimitiveEllipse, PrimitiveLine... added as constants to the class :py:class:`plotItem` of itom module in Python.
* double click on Python error message (filename, line-number) in console opens the script at the indicated line
* increased buffer size of data object protocol strings since sprintf crashes upon buffer overflow (even sprintf_s, debug assertion)
* Improved :py:meth:`itom.plotHelp`-command to get a list of all plot widget
* new demo scripts added (see demo subfolder)
* :py:meth:`pointCloud.copy()` for deep copies added
* Make Python loops interruptible from GUI
* Added functions to edit, add and read out color maps bases on rgba-values
* :py:meth:`itom.addMenu` / :py:meth:`itom.removeMenu` / :py:meth:`itom.addButton` / :py:meth:`itom.removeButton`: the methods to add a button or menu element always return an unique handle to the recently added item. The methods to remove them can be called with this handle to exactly delete the element that has been added (does not remove an element with the same name that replaced the previously added one). With this change, a bug has been fixed: In some cases methods or functions connected to menu items have not been released when deleting the parent menu element.
* observeInvocation in AbstractAddInConfigDialog and abstractAddInDockWidget now use the waitAndProcessEvents method of ItomSharedSemaphore instead of a "self-programmed" version doing the same.
* :py:meth:`itom.addButton` returns handle to button, this handle can be used to call :py:meth:`itom.removeButton`(handle) to exactly delete this button. The existing method :py:meth:`itom.removeButton`(toolbarName, buttonName) still exists as other possibility
* AddInManager::initAddIn: let this method also processEvents when it waits for the plugin's init method to be finished. Then, the plugin can directly invoke slots from GUI-related elements of the plugin within its init method. * Bugfix in dataObject.toGray() with default parameter (type)
* QPropertyEditor can now handle QStringList properties (with list editor dialog for editing the QStringList)
* methods removeItem, setItemData and insertItem of QComboBox are now callable from Python (via .call(...) command)
* Qt5 fixes and adaptions (encodings, qmessagehandler, linkage against shell32 for Qt5.4 and Windows, ...)
* 'log' can be passed as argument to the executable to redirect debug outputs to itomlog.txt.
* always set Environmental Variable MPLCONFIGDIR to itom-packages/mpl_itom such that an individual matplotlibrc config file can be placed there. You can set the backend within this config file to module://mpl_itom.backend_itomagg such that matplotlib always renders to itom windows if used within itom.
* All-In-One build environment for Qt5, Python 3.4, VS2010, 32 or 64bit, PCL 1.8.0, VTK 6.1 created. Documentation about all-in-one build environment added.
* Added more flexible thread-safe control classes for all types of dataIO (grabber, ADDA, rawIO) and actuators (using inheritance).
* added multi-parameter slot getParamVector to ito::AddInBase (similar to setParamVector).
* class CameraThreadCtrl inserted in helperGrabber.h (part of SDK; replacement for threadCamera). The new class guards the camera by temporarily incrementing its reference counter, checks the isAlive() flag when waiting for semaphores and directly frees the semaphore after usage. A camera pointer can directly be passed to this class.
* some checks inserted to avoid crashes at startup if Python could not be found.
* theme for html documentation adapted
* replaced c-cast in REMOVE_PLUGININSTANCE by qobject_cast (due to some rare problems deleting plugins)
* documentation: sphinx extension Breathe updated to version 3.1
* improved parameter validation for floating point parameters (consider epsilon value)
* typedef ParamMap and ParamMapIterator as simplification for QMap<QString,ito::Param> ... inserted
* better detection of OpenCV under linux
* fix in backend of matplotlib (timer from threading package to redraw canvas for instance after a zoom or home operation did not work. Replaced by itom.timer)
* range: (min,max,step) and value: (min,max,step) also added for rangeWidget of itomWidgets project. rangeSlider and rangeWidget can now also be parametrized passing an IntervalMeta or RangeMeta object.
* RangeSlider widgets now have the properties minimumRange, maximumRange, stepSizeRange, stepSizePosition
* some new widgets added to itomWidgets project
* IntervalMeta, RangeMeta, DoubleIntervalMeta, CharArrayMeta, IntArrayMeta, DoubleArrayMeta, RectMeta added as inherited classes from paramMeta. AddInInterface incremented to 1.3.1.
* bugfix in dataObject when creating a huge (1000x500x500, complex128) object (continuous mode). 
* method :py:class:`uiItem.exists` added in order to check if the widget, wrapped by uiItem, still exists.
* bugfix in squeezeConvertCheck2DDataObject if numberOfAllowedTypes = 0 (all types allowed)
* Python script navigator added to script editor window (dropdown boxes for classes and methods)
* implicit cast of ito::Rgba32 value to setVal<int> of ito::Param of type integer.
* improved integration of class itom.autoInterval and the corresponding c++ class ito::AutoInterval. Conversion to other data types added.
* documentation about programming of AD-converter plugins (dataIO, type ADDA) added
* several bugfixes in breakPointModel (case-insensitivity of filenames,  better error message if breakpoint is in empty line, ...)
* many other bugs fixed
* improvements in documentation


**Version 1.3.0 (2014-10-07)**

(more than 150 commits in itom repository)

* fixes big bug in assignment operator of dataObjects if d2=d1 and d1 shares its data with an external object (owndata=0). In the fixed version, d2 also has the owndata flag set to 0 (before 1 in any cases!)
* replace dialog for replacing text within a selection fixed
* API function apiValidateAndCastParam added. This is an enhanced version of apiValidateParam used in setParam of plugins. The enhanced version is able to modify possible double parameters in order to fit to possibly given step sizes.
* support of PyMatlab integrated into CMake system (check BUILD_WITH_PYMATLAB to enable the python module 'matlab' and indicate the include directory and some libraries of Matlab)
* Add twipOGL-Plugin (from twip Optical Solutions GmbH) to plotToolBar if present
* dataObject: axisUnits, axisDescriptions, valueUnit and valueDescription can now handle encoded strings (different than UTF8)
* QDate, QDateTime and QTime are now marshalled to python datetime.date, datetime.datetime and datetime.time (useful for QDateTimeEdit or QDateEdit widgets)
* text from console widget is never deleted when drag&dropping it to any script widget
* state of python reloader is stored and restored using the settings
* warning is displayed if figure class could not be found and default fallback is applied
* :py:func:`itom.plotLoaded` and :py:func:`plotHelp` added
* Improved protocol functionality for dataObject related python functions
* Fixed missing copy of rotationMatrix metadata for squeeze() function in dataobj.cpp
* Added copyAxisTagsTo and copyTagMapTo to ito::absFunc, ito::realFunc, ito::imagFunc, ito::argFunc to keep dataTags
* example about modal property dialogs added to demo folder
* QScintilla version string added to version helper and about dialog
* :py:func:`itom.clc` added to clear the command line from a script (useful for overfull command line output)
* AutoInterval class published in common-SDK. This class can be used as datatype for an min/max-interval (floats) with an additional auto-flag.
* public methods **selectedRows**, **selectedTexts** and **selectRows** of QListWidget can now be called via :py:meth:`itom.uiItem.call`
* itom.dataObject operator / and /= for scalar operand implemented (via inverse multiplication operator) fixed casting issue in multiplication operator for double scalar (multiply with double precision -> then cast)
* configured QPropertyEditor as shared library (dll) instead of static library. This was the last library that was not shared yet.
* reduced size of error messages during live image grab
* fix when reloading module, that contains syntax or other errors: these errors are displayed for better handling
* If an instance of QtDesigner is opened and then an ui-file is loaded, this file was opened in a new instance of QtDesigner. This is fixed now.
* some crashes when not starting with full user rights are fixed
* Added demofile for data fitting
* if last tab in script window is closed, the entire window is closed as well
* types ito::DataObject, QMap<QString, ito::Param>... are no registered for signal/slot in addInManager such that this needs not to be done in plugins any more.
* class **enum** in *itomPackages* renamed to **itomEnum** since Python 3.3 introduces its own class **enum**.
* check for AMD-CPUs and adjust environment variable for these processors in order to avoid a KMP_AFFINITY error using Matplotlib (and OpenMP).
* enhanced output for class function :py:meth:`itom.ui.info`.
* optional properties argument added to commands :py:func:`itom.plot` and :py:func:`itom.liveImage`. Pass a dictionary with properties that are directly applied to the plot widget at startup.
* recently opened file menu added to itom main window and script windows
* improved loaded plugins dialog
* some fixes in data object: fixed constructor for existing cv::Mat and type RGBA32, fixed bug in assignment operator for rhs-dataObjects that do not own their data block.
* property dialog documented
* improved python module auto reloader integrated into itom (based on iPython implementation). This reloader can be enabled by the menu script >> reload modules.
* some examples from the matplotlib gallery added to the demo scripts
* bugfix when changing the visibility of undocked plots
* the designer plugin *matplotlib* has now the same behaviour than other plot widgets (can be docked into main window, property toolbox available...)
* some improvements with tooltip texts displaying optional python syntax bugs (python module frosted required)
* unified signatures passed to Q_ARG-macro of Qt.
* the search bar is not hidden again if Ctrl-F is pressed another time.
* detailed descriptions of plugins are now also displayed in help toolbox
* improvements to reject the insertion of breakpoints in commented or empty lines
* improved breakpoint toolbox that allows en-/disabling single or all breakpoints, deleting breakpoints... Breakpoints are reloaded at new startup.
* unused or duplicated code removed and cleaned
* project *itomWidgets* synchronized to changes in mother project commonCTK 
* german translations improved
* itom and the plugins now support a build with Qt5. The setup is still compiled with Qt4.8.
* support for CMake 3.0 added

**Version 1.2.0 (2014-05-18)**

(more than 300 commits in itom repository)

* :py:meth:`itom.dataObject.toGray` added in order to return a grayscale dataObject from type rgba32.
* Drag&drop python files to command line in order to open them
* :py:meth:`itom.actuator.setInterrupt` added in order to interrupt the current movement of an actuator (async-mode only, general interrupt must be implemented by plugin)
* Window bugfix (improper appearance) in Qt5 (windows) as well as some linux distributions (Qt4/Qt5)
* Breakpoint toolbox improved.
* Open scripts and current set of breakpoints is saved at shutdown and restored at new startup.
* Python, Numpy, Scipy, Matplotlib and itom script references can be automatically downloaded and updated from an itom online repository. These references can then be shown in the itom help toolbox.
* CMake: Improvements in find_package(ITOM_SDK...) COMPONENTS dataObject, pointCloud, itomWidgets, itomCommonLib and itomCommonQtLib can be chosen. Since dataObject and pointCloud require the core components of OpenCV and PCL respectively, these depending libraries are detected as well and automatically added to ITOM_SDK_LIBRARIES and ITOM_SDK_INCLUDE_DIRS.
* :py:class:`itom.npDataObject` removed (reasons see: http://itom.58578.x6.nabble.com/itom-discussions-deprecated-class-itom-npDataObject-td3.html)
* Designer plugins can now display point clouds and polygon meshes (itomIsoGLFigurePlugin displays pointClouds)
* improved codec handling. Default codec is latin1 (ISO 8859-1) now, Python's codec is synchronized to Qt codec.
* Modified / new templates added for creating a grabber, actuator and algorithm plugin.
* ito::DataObject registered as meta type in itom. Plugins do not need to do this any more.
* Styles in property editor can be reset to default, size of all styles can be globally increased or decreased
* C-Api: new base classes for plugin configuration dialogs and dock widgets added to SDK: ito::AbstractAddInDockWidget and ito::AbstractAddInConfigDialog
* modified and improved user documentation
* itom extension added for sphinx for semi-automatic plugin documentation
* More CMake macros added for easier implementation of plugins: FIND_PACKAGE_QT, PLUGIN_TRANSLATION, PLUGIN_DOCUMENTATION
* Python commands :py:func:`itom.pluginHelp` and :py:func:`itom.filterHelp` either return dict with information or print information (depending on arguments)
* Plugin can provide a rst-based documentation file. The documentation of all available plugins is then integrated in the itom help system.
* Python syntax check in script editors introduced. Install the python package **frosted** in order to obtain the syntax check (can be disabled in property dialog)
* meta information stepSize introduced for ito::IntMeta and ito::DoubleMeta
* glew is not necessary if building against Qt5 with OpenGL.
* New shared library **itomWidgets** added to SDK. itom, user defined GUIs and plugins have now access to further widgets. These widgets are mainly taken from the CTK project (the common toolkit).
* Python class **enum** in itom-packages renamed to **itomEnum** since a class enum (with similar functionality is introduced with Python >= 3.4)
* All signaling_NaNs replace by quiet_NaNs (signaling_NaNs raise C exception under certain build settings)
* Unittests for ito::ByteArray, ito::RetVal and ito::Param added
* User management partially available via Python.
* itom can now be build and run using Qt4 or Qt5. Usually the Qt-version installed on the computer is automatically detected, if both versions are available use the CMake variable BUILD_QTVERSION and set it to Qt4 or Qt5 for manual choice.
* Help toolbox also shows information about all loaded hardware plugins (actuator, dataIO)
* Help toolbox in itom shows information about filters and widgets provided by algorithm plugins. Exemplary code snippet for all filters added.
* itom can be build without PCL support, hence the point cloud library is not required. The library *pointCloud* is not available then (see BUILD_WITH_PCL in CMake)
* C-Api: New shared libraries created: itomCommon (RetVal, Param,...), itomCommonQt (ItomSharedSemaphore, AddInInterface,...), dataObject, pointCloud, itomWidgets. Plugins link agains these shared libraries. This allows an improved bugfix in these components. Additionally, many changes in these libraries don't require a new version number for the plugin interface (unless the binary compatibility is not destroyed).
* C-Api: error message in ito::RetVal, name and info string of ito::Param are now stored as ito::ByteArray class that is based on a shared memory concept. Less memory copy operations are required.
* crash fixed if itom is closed with user defined buttons and menus
* fixes if some components are disabled by user management
* C-Api: DesignerPlugins can now be used within other plugins (e.g. widgets in algorithm plugins)
* many bugfixes

**Version 1.1.0 (2014-01-27)**

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

**Version 1.0.14 (2014-09-02)**

there is no continuous changelog for these version

Plugins
******************

**Version 1.4.0 (2015-02-17)**

(more than 170 commits in plugins repository)

* vertex and fragment shader error in gl based plugins: Since NVIDIA 347.xx, no character (including space, \n...) must be before #version directive. Else shaders may not be compiled (error C0204, version directive must be first statement)
* linux support for ttyS, ttyUSB and ttyACM in plugin **serialIO** added, selection via portnumber S=0-999 USB=1000-1999 ACM=2000-2999
* some fixes in **FireGrabber**
* fix a crash in **PCOPixelFly** if camera is not correctly connected with the board
* fix in **OpenCVGrabber** when using the EasyCAP or similar analog-digital video converters (USB2)
* filters *cvResize*, *cvSplitChannels*, *cvMergeChannels* added to **OpenCVFilters** for interpolated size change of a data object
* timestamp added to tags of acquired data objects in **PGRFlyCapture**. Further fixes to achieve highest possible framerate. (don't use extended shutter for high frame rates and don't set the smallest possible frame time, since this leads to a really low frame rate, a frame time close to the minimum achieves high frame rates.)
* **Ximea**: bugfixes, added *gpo* and *gpi*, backward compatibility to older Ximea API...
* filter *cropBox* for **PclTools** inserted to crop points inside or outside of an arbitrary positioned and rotated box.
* **PclTools** adapted to PCL 1.8 and split into several files due to a big-object-compiler error.
* bugfixes in **glDisplay**, can now also display Rgba32 data objects. Additionally, existing textures can be changed with *editTextures*.
* many config dialogs and toolboxes inherit now the new abstract base classes.
* fixes in grabFramebuffer of **dispWindow** plugin
* fixes in cvUndistortPoints, cvInitUndistortRectifyMap and cvRemap (**OpenCVFilters**)
* fitPlane and getInterpolatedValues in **FittingFilters** can now also be executed using the "least median of squares"
* plugin **OpenCVFiltersNonFree** created. This contains non BSD licensed code from OpenCV and is not included in the itom setup per default.
* plugin **AVTVimba** created and released under LGPL (for cameras from Allied Vision)
* plugin **AndorSDK3** create and released under LGPL (for cameras from Andor, tested with Zyla 5 camera)
* plugin **NewportSMC100** added and released under LGPL to control actuators from Newport (SMC 100)
* plugin **libmodbus** added and released under LGPL. This supports the communication protocol *modbus* (based on libmodbus 3.1.2 from https://github.com/stephane/libmodbus)
* plugin **PI_GCS2** added and released under LGPL. This controls *Physik Instrumente* devices using the GCS2 command set (tested with E-753).
* plugin **demoAlgorithms** released under LGPL.
* plugin **SuperlumBS** added and released under LGPL (for Broadband swept light source).
* plugin **NI-DAQmx** for National Instruments DAQmx interface added and released under LGPL.

**Version 1.3.0 (2014-10-07)**

(more than 100 commits in plugins repository)

* plugin *Thorlabs CCS* for spectrometers from Thorlabs added (dataIO plugin). This plugin requires further drivers from the Thorlabs device.
* plugin *AerotechA3200* added to support the deprecated A3200 interface from company Aerotech.
* fixes in plugin *PIPiezoCtrl*: parameters delayOffset, delayProp and async are now really transmitted to the device (did nothing before)
* fixes in *PCOCamera* plugin with camera *PCO.1200s* that does not support the setPoint temperature.
* all plugins adapted for Qt4 and Qt5.
* plugin *dispWindow* adapted to OpenGL 3.1 and 4.0. Deprecated shader commands replaced. Parameters 'lut' and 'gamma' are now working and the gamma correction is enabled if parameter *gamma*=1
* filter *cvUndistort* in *OpenCVFilters* can now handle every data type as input.
* fixes some bugs when importing csv files
* filter *cvFlipFilter* also supports multi plane flipping for 3D data objects.
* plugin *GLDisplay* added that allows displaying one or multiple arrays on the screen using OpenGL to provide a very fast flip between several images.
* many enhancements and improvments in plugin *pclTools* (mainly done by company twip optical solutions GmbH): filter for fitting spheres to point clouds added, filter to calculate distances to a given model added, filter to prepare a display of these distances added, methods partially OpenMP parallelized, filter for fitting cones to point clouds added, filter for projecting point clouds to models added.
* plugin *PGRFlyCapture* now runs under linux, general changes to support Grasshopper3 cameras (color supported as well).
* some fixes in plugin *cmu1394* and optional byte swapping for 16bit camera added
* improvements in camera plugin *Vistek*
* improved error handling when trying to load unsupported tiff formats
* filters *gaussianFilter* and *gaussianFilterEpsilon* added to plugin *BasicFilters*
* filters *cvRotate180*, *cvRotateM90* and *cvRotateP90* added to *OpenCVFilters*
* improvements and better synchronization to camera in plugin *Ximea*. Experimental shading correction added.
* bugfix when loading a x3p file -> yscale has not been loaded correctly
* camera plugin *IDS uEye* added to support cameras from company IDS Imaging (based on their driver 4.41)

**Version 1.2.0 (2014-05-18)**

(more than 200 commits in plugins repository)

* PCOCamera: improved plugin with renew config dialog and toolbox. Tested with PCO.1300 and PCO.2000.
* DataObjectIO: Added importfilter for ascii based images or point lists
* PCLTools released under LGPL
* PGRFlyCapture: extended shutter added for longer integration times, frame rate is not used if extended shutter is one since images are acquired as fast as possible.
* AerotechEnsemble: bugfixes: avoid timeout for long-time operations corrected status message when all axes are in-target position
* fixes for PCOPixelFly: some board IO errors are handled
* DispWindow adapted to OpenGL 3.x specification as well as Qt5
* Initial release of video 4 linux (V4L2)
* Improved SDF-Export function with invalid handling for MountainsMaps (plugin DataObjectIO)
* exec-function for ramp-trajectory added to GwInstek plugin (power supply controller with RS232 connection)
* Plugin FireGrabber supported under linux. FirePackage driver used under Windows, Fire4Linux under linux
* Plugin SerialIO under linux: both tty and usb ports supported
* Plugin x3pio released under LGPL (wrapper for data format x3p, see opengps.eu)
* Added clipping-Filters and history-filter to BasicFilters
* Many documentations for plugins added
* Plugin MSMediaFoundation released under LGPL (requires at least Windows Vista, successor of DirectShow for accessing basic cameras)
* Parameters sharpness and gamma added to Ximea plugin.
* Plugin LibUSB released.
* Fixed big-endian, little-endian bug in PointGrey plugin, parameters sharpness, gamma and auto_exposure added
* Plugin for PointGrey cameras released under LGPL.
* Plugin for XIMEA cameras released under LGPL.
* FFTW plugin added that is a wrapper to the FFTW library (GPL license!)
* overall modification of Vistek camera plugin: Toolbox, configuration dialog based on new base classes of itom SDK, better parameter handling, improved image grabbing...

**Version 1.1.0 (2014-01-27)**

there is no continuous changelog for these version

Designer Plugins
******************

**Version 1.4.0 (2015-02-17)**

(more than 50 commits in designerPlugins repository)

* itom1dqwtplot: changed slots for setPicker / getPicker from ito::int32 / ito::float32 to int and float due to conversion / call problems.
* itom1dqwtplot: fixes some rescaling problems when switching the complex representation or the row/column representation.
* itom1dqwtplot, itom2dqwtplot: improvements in panner, magnifier, zoomer with and without fixed aspect ratio. Magnification is now possible using Ctrl + mouse wheel.
* itom1dqwtplot, itom2dqwtplot: geometric elements can now obtain labels (accessible via slots)
* Initial commit of vtk3dVisualizer to visualize pointclouds, polygon meshes, geometric elements. These elements are organized in a tree view and can be parametrized. The display is realized using Vtk and the PointCloudLibrary.
* Encoding fixes in itom1dqwtplot and itom2dqwtplot due to default encoding changes in Qt5 (switched from Latin1 to Utf8)
* itom2dqwtplot:  Added property to change geometric element modification mode
* itom1dqwtplot:  Improved linewidth for copy pasted export
* itom1dqwtplot, itom2dqwtplot:  zoom stack of zoomer and magnifier tools is synchronized with panner such that changing the plane or complex representation does not change the zoomed rectangle after a panning event)
* itom1dqwtplot, itom2dqwtplot: some handling fixes in export properties of 1d and 2d qwt plot. The properties are now shown before the file-save-dialog in order to give the user an overview about the possibilities before he needs to indicate a filename.
* itom1dqwtplot, itom2dqwtplot: shortcuts added for actions 'save' and 'copyToClipboard'
* itom1dqwtplot: property lineStyle and lineWidth added
* itom1dqwtplot, itom2dqwtplot: copy-to-clipboard added to tools menu of 1d and 2d qwt plot. Improved keyPressEvents for both plots (playing with event->ignore() and event->accept())
* itom1dqwtplot: rounding fix in order to show the right data to given z-stack-cut coordinates.
* improvements in itom2dqwtplot: z-stack picker obtains the general color set including a semi-transparent background box; the z-plane can be selected via a property 'planeIndex'
* itom2dqwtplot: z-stack and linecut window has an appropriate window title
* itom1dqwtplot, itom2dqwtplot: Working on an improved geometric element handling (e.g. modes for move, modify points) Adapted type switches and comparisons to handle flagged geometric elements via type ito::PrimitiveContainer::tTypeMask 
* itom1dqwtplot, itom2dqwtplot: Added new icons for geometric element modification.
* Added shift and alt modifier to itom2dqwtplot to move / rotate geometric lines with fixed length
* update to qwt 6.1.2 for compability with Qt 5.4
* Improving EvaluateGeometricsFigure to evaluate 3D-Data
* Improved functionality of EvaluateGeometricsFigure to calculate distances between ellipse centers
* fix in itom1dqwtplot and itom2dqwtplot: dataObjects were not updated if only their content, but not the size, type... changed
* changes for access of plotItemChanged via python
* Added colorMap to overlayImage for Itom2dQwtPlot via overlayColorMap-Property
* itom1dqwtplot: legend (optional) added to itom1dqwtplot (properties: legendPosition (Off, Left, Top, Right, Bottom) and legendTitles (StringList) added). Per default, the legend is not displayed, and if it is displayed, the default names are curve 0, curve 1, curve 2...
* itom2dqwtplot is principally able to display 1xN or Nx1 data objects (was blocked until now; but sometimes people want to do this)
* itom1dqwtplot, itom2dqwtplot adapted to ito::AutoInterval class (xAxisInterval, yAxisInterval, zAxisInterval are of this type now)
* itom1dqwtplot, itom2dqwtplot: Added background, axis and tick color

**Version 1.3.0 (2014-10-07)**

(more than 40 commits in designerPlugins repository)

* The properties *xAxisInterval* and *yAxisInterval* return the currently visible or set interval (bugfix)
* overlay image of *itom2dqwtplot* can now be read out by the property *overlayImage*
* Firsts steps for an auto-documentation of designer plugins
* Linecut of *itom2dqwtplot* can now also be set to the horizontal / vertical line containing the global minimum or maximum
* Some bugfixes concerning the display of dataObjects that are shallow copies from numpy arrays
* Fixes a bug that showed errors when a linecut and a z-stack-cut of *itom2dqwtplot* and a 3d data object is visible at the same time
* Mode for single and multi row or column display of *itom1dqwtplot* for 2d data objects as input
* Center marker in *itom2dqwtplot* can be adjusted in size and pen using the general style settings for designer plugins (itom.ini setting file only)
* improvements and rework of zoomer, panner and magnifier with or without fixed aspect ratio for *itom2dqwtplot* and *itom1dqwtplot* 
* magnifier of *itom2dqwtplot* and *itom1dqwtplot* now also works with Ctrl+mousewheel
* pickPoints event now also works in *itom2dqwtplot* if the zoomed rectangle or the magnification is changed during interaction
* improved unit switches in GUI in *motorController*
* slots added for saving and rendering to pixmap for *itom2dqwtplot*
* property *grid* added to *itom1dqwtplot* to show/hide a grid
* some problems fixed with point selectors in *itom1dqwtplot*
* matplotlib is now a designer plugin based on AbstractFigure like other plots as well. It can then be docked into the main window.

**Version 1.2.0 (2014-05-18)**

(more than 80 commits in designerPlugins repository)

* *itom2DQwtPlot* is able to display color data objects and cameras.
* Press Ctrl+C to copy the currently displayed plot in *itom1DQwtPlot* and *itom2DQwtPlot* to clipboard. Also available via menu.
* display of pointClouds in *itomIsoGLFigurePlugin*
* fix in autoColor mode (*itom2DGraphicView*) with rgba32 data objects or cameras
* save dialog remembers last directory
* secondary dataObject can be plotted as semi-transparent overlay (alpha value adjustable) in *itom2DQwtPlot* (Python access via property *overlayImage*)
* many bugfixes
* multiline plotting in *itom1DQwtPlot* improved
* designer plugins are now ready for inclusion in GUIs of other plugins

**Version 1.1.0 (2014-01-27)**

there is no continuous changelog for these version
