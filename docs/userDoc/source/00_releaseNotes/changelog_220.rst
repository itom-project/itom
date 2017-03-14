.. include:: ../include/global.inc

.. |mm| unicode:: U+00B5 m

Changelog Version 2.2.0
#########################

itom
********

**Version 2.2.0 (2016-10-01)**

(more than 500 commits in itom repository)

* Class :py:class:`~itom.figure` now provides the optional parameters x0,y0,width and height to initially position the figure. This is e.g. used by matplotlib.
* added :py:meth:`itom.dataObject.splitColor` function. Now the colors of a rgba dataObject can be splitted.
* ito::PCLPointCloud::copy() and ito::PCLPointCloud::scaleXYZ(...) added to pointClouds library. Method itom.pointCloud.scaleXYZ(...) added to Python class.
* bugfix in dialogOpenFileWithFilter when loading Rgba32 dataObjects due to char-array access out of range for Rgba32
* mat and idc files can be loaded as packed or unpacked version in workspace
* If a RuntimeWarning is now displayed in the command line, a double click will open the file at the indicated line
* bugfix crash in dataObjectIterator
* improved message if numpy is tried to be upgraded via pip manager. If pip manager is started as standalone application, itom can be started afterwards from the dialog.
* Dialog added to scripts menu to show all currently active :py:class:`~itom.timer` instances. They can be stopped and restarted from this dialog.
* improved roughnessEvaluator tool to provide a GUI for the 1D roughness evaluation
* statusLED widget added to itomWidgets
* :py:func:`itom.dataObject.fromNumpyColor` and :py:func:`itom.dataObject.toNumpyColor` added to load and write coloured numpy arrays (3 dimensions with last dimension of size 3 or 4) to a rgba32 data object. This is interesting if objects are processed using cv2 (OpenCV) in Python.
* generic snapshot dialog added that is available for all camera instances (e.g. by their context menu)
* improved drag&drop of files to the console. Hourglass is shown as cursor while a file is loaded.
* documentation added how to directly start the package manager with an argument to qitom.exe
* bugfix in matrix multiplication operator of DataObject: The data object has to be reallocated both for :literal:'*' and :literal:'*'= operators if the size of the multiplied matrix is different than the size of
* method meanValue implemented in dataObjectFuncs using the quick median algorithm.
* added available colorMaps in plot documentation
* maximum number of threads that should be used for algorithms and filter plugins can be set in property dialog. In plugins use ito::AddInBase::getMaximumThreadCount() (static) to request this value. It is always bound between 1 and the maximum number of avaiable threads on the computer.
* context menu of command line reimplemented to provide itom-specific commands and respect that executed command cannot be removed any more
* redesign of matplotlib backend to enable faster rendering and updates only if necessary
* Proper installation of Python is checked at startup of itom. If Python Home directory could not be properly resolved (e.g. its search pathes do not contain the script 'os.py' as built-in script, an error is set and python is not started. The user can define a user-defined PYTHONHOME variable in the property dialog of itom.
* added Dialog for opening various files at once (:py:func:`~itom.ui.getOpenFileNames`)
* dataObjects (as well as numpy.arrays), polygonMeshes and pointClouds can now be plot from context menu of corresponding variables in workspace toolboxes
* mainWindow: menu "View" shows now also a list of all opened scripts
* == and != operator added for ito::ParamBase. This checks if types are equal and if so, if the content is equal. Double and complex values (single and arrays) are checked using their corresponding epsilon-tolerance.
* CMake Option BUILD_OPENMP_ENABLE inserted to all plugins. If this is disabled, the pre-compiler USEOPENMP is disabled, even if a release build is started and OpenMP is available. Plugins can now get the maximum number of cores by static in AddInBase::getMaximumThreadCount(). This value can be limited by the user in the ini-setting file. Background: Sometimes, it is better to not use all available threads.
* user dialog: always pre-selects the user profile with the name of the currently logged user or - if not available - the last used profile
* slots with enumeration types can be called now (only Qt5). For Qt < 5.5, only integer values can be passed; later also the name of the desired enumeration types.
* Method :py:func:`~itom.getQtToolPath` added
* Improved support of style sheets in itom. Default stype sheets can be set in property editor. Styles of command line and scripts can now also be fully adjusted via property editor.
* checkableComboBox of itomWidgets now has the property 'checkedIndices' to get a list of indices which are checked or set them.
* added new generic widget *motorAxisController* to itomWidgets library. This widget can control any actuator plugin, either in any GUI or the internal toolbox of the plugin. Due to this, the library *commonQtLib* was split into *commonQtLib* and *commonPlotLib*
* command line is not automatically scrolled to the end of the text if the mouse wheel is turned. The automatic scrolling is re-enabled once a new command is entered or the command line is cleared
* returned handles of itom.plot or itom.liveImage now also have the outer window as baseItem such that connect commands to these plots are successful
* script plot_help_to_rst_format.py added to convert the output of itom.plotHelp(...) to a rst-string format
* sphinx extension 'numpydoc' updated
* ensure that :py:func:`~itom.saveIDC` (pickle.dump) always saves with the defined protocol version 3 (default for Python 3).
* better auto-documentation of designer plot widgets, parsed in the C++ or python context. Slots marked with the ITOM_PYNOTACCESSIBLE macro are excluded from the Python context.
* From Qt 5.6 on, itom is shipped with a built-in help viewer based on QtWebEngineView. QtAssistant is not used any more for this purpose.
* Added method :py:func:`~itom.showHelpViewer` to open help viewer
* PythonEngine: added slots pickleSingleParam and saveMatlabSingleParam to save objects to idc and mat by a C++ code
* many settings of command line and scripts can now be adjusted via property dialog (e.g. spacing between lines)
* many docstrings of Python methods improved
* documentation about build for raspberry pi added
* linux fixes for FindQScintilla: able to detect package libqt5scintilla
* ito::areEqual added to numeric.h in commonLib
* pip manager: added message under windows that some packages depend on certain versions of the MSVC redistributable package
* error messages emitted by debugger (e.g. cannot add breakpoint...) can now also be parsed by double click event in command line such that indicated script is opened.
* ItomUi: autoslot also possible with window or dialog itself (by its objectName)
* typeDefs.h reduced to values that are important both for the core and plugins. Core-only enumerations have been moved to global.h
* plugins and filters can now also provide parameters of type *complex* or *complex array*
* auto grabbing interval (in ms) for grabbers can now be read and adjusted by :py:meth:`~itom.dataIO.getAutoGrabbingInterval` and :py:meth:`~itom.dataIO.setAutoGrabbingInterval`. Better error messages inserted in startDevice fails if a live image is created.
* AddInGrabber: Auto grabbing is disabled if errors occurred consecutively during the last 10 runs.
* user documentation extended
* added AutoGrabbing to toolbar
* further path variables can now either be appended or prepended to the global PATH variable (see property dialog)
* Init-Dialog of plugin instances now also provide a GUI for array parameters (e.g. integer or float array)
* Python Matlab engine now dynamically resolves functions from libeng.dll and libmx.dll of Matlab. Therefore, itom can be build with Matlab support and matlab must not necessarily be available on the destination computer.
* documentation added, how to create the all-in-one development setup
* frosted syntax check can now also detect the usage of an unreference variable.
* added optional parameter *parent* to :py:meth:`~itom.ui.getText`, :py:meth:`~itom.ui.getInt`, :py:meth:`~itom.ui.getDouble`
* bugfix: itom crashed by calling :py:meth:`~itom.dataObject.pixToPhys` with negativ axis
* improvements and fixes concerning interpreter lock GIL of Python
* :py:meth:`itom.font.isFamilyInstalled` and :py:meth:`itom.font.installedFontFamilies` added to :py:class:`itom.font`
* modifications in PaletteOrganizer, e.g. add of new matplotlibs colormap viridis and improved line colors
* new class *QPropertyHelper* added to C++ sources to bundle all conversions between Q_PROPERTY properties and other Qt classes.
* fixes in itomWidgets, partially due to commits to the original ctkCommon project, colorPickerButton and colorDialog added to itomWidgets (taken from ctkCommon project)
* Python type None can now be assigned to slots of type QVector<ito::Shape>, QSharedPointer<ito::DataObject>, ...
* workaround in doubleSpinBox of itomWidgets for Qt < 5.5 such that the minimum size of the spinbox will not be huge if for instance its minimum or maximum value is +/- Inf.
* ito::DataObject::bitwise_not added to create a bitwise inversion. This method is also connected to the ~operator of :py:class:`itom.dataObject`
* param validator can now parse the name of the parameter in error messages
* improved range check for 'areaIndex' added to :py:meth:`itom.figure.plot`, :py:meth:`itom.figure.liveImage` and :py:meth:`itom.figure.matplotlibFigure`
* bugfix in mapping set of itom.dataObject if a numpy-array with the same squeezed shape but different unsqueezed shape is given.
* static methods of :py:class:`~itom.ui` (like :py:func:`itom.ui.msgInformation`) can obtain uiItem, ui, figure or plotItem as parent argument
* matplotlib designer widgets are now handled like any other plot. They can be docked or stacked into sub-figures. The command :py:meth:`~itom.close` also closes matplotlib figures.
* many demo scripts added to demo folder and its subfolders (e.g. plots, shapes, matplotlib...)
* more checks and features added at startup to better improve foreign text codecs (e.g. Russian). Bugfix in PythonEngine::stringEncodingChanged()
* ito::pclHelper::dataObjToEigenMatrix added
* settings of plots are not only read from ini-setting file but are also tried to be inherited from parent plot (e.g. 1D plot as child of 2D plot)
* first plot / figure get the number 1 instead of 2.
* Fixes to display WaitCursor during long operations (e.g. file load)
* bugfix if a dataObject is freed in special cases.
* improved help rendering in plugin help toolbox.
* dataObject.zeros, ones, eye, randN and rand returned int8 as default type, but the documentation said uint8 (like the default constructor). This was a bug. It is fixed now, all static constructors return uint8 per default.
* Improvements for displaying help about plugins (e.g. in command line, added scripts to semi-automatically render help for documentation...)
* methods *isNotZero*, *isFinite*, *isNaN* and *isInf* moved from dataobj.h to numeric.h of Common library
* :py:meth:`~itom.dataObject.createMask` added to create a mask object from one or multiple shapes.
* enumerations based on QFlags (bitmask) can now be set via property in Python. These properties are now also changeable in property editor toolbox.
* SDK adapted to support shapes in plots (rectangles, circles, points...). These shapes are accessible via :py:class:`itom.shape` and defined in Shape library. Old-style primitives (dataObject based) in plots replaced by shapes.
* Implementation for generic toolboxes for markers, pickers and shapes of plots
* many bugfixes


Plugins
******************

**Version 2.2.0 (2016-10-01)**

(more than 160 commits in plugins repository)

* all plugins: ito::dObjHelper::isXYZ replaced by ito::isXYZ (e.g. isFinite)
* PclTools: fix in *pclRandomSample* to avoid large subareas without selected samples
* PclTools: filters *meshTransformAffine* and *pclTrimmedICP* added to apply a coordinate transform to a polygonal mesh
* PclTools: EIGEN2_SUPPORT define removed from PclTools since unused. If this symbol is defined, newer Eigen libraries cannot be used.
* FirgelliLac: Plugin added for stepper motor from Firgelli
* Ximea: plugins compiles now under linux
* Ximea: built with API version 4.10.0, fixes inserted for using SDK 4.06 and 4.04
* Ximea: disable gammaColor parameter for monochrome cameras, plugin is ready to used color cameras now
* PGRFlyCapture: timestamp bugfix for very long acquisitions in PointGrey FlyCapture
* OpenCVGrabber: can now also be opened with a video file or stream url
* MSMediaFoundation: fixes mainly for controlling the integration time (in seconds)
* IDSuEye is now shipped with driver 4.80.2 (Windows) and compiles under linux
* IDSuEye: bugfix in max. roi size with newer cameras, extended integration time mode set to readonly if not available
* ThorlabsDCxCam: plugin initially pushed to support IDS OEM cameras from Thorlabs. This plugin is very similar to IDSuEye, but has not finally been tested, yet.
* AVTVimba: build with SDK version 2.0
* FittingFilters: filter *subtractRegressionPlane* has now the same parameters than *fitPlane*
* FittingFilters: new filter *fillInvalidAreas* added to fill small invalid areas in topography data using interpolation based on all surrounding values.
* NanotecStepMotor: compiles now under Linux and Windows. init parameter 'axisSteps' has no step size constraint for each axis.
* x3pio: bugfix if xyUnit or valueUnit is set to '|mm|': wrong detection of |mm| string due to encoding problems
* x3pio: fix in x3p for saving line based data
* BasicFilters, OpenCVFilters, DataObjectArithmetic and others: docstrings of filters improved
* BasicFilters: filter *sobelOpt* added for improved version of sobel filter (named Scharr filter)
* BasicFilters use now the global setting for the maximum number of threads for OpenMP parallelization
* BasicFilters: bugfix for lowPassFilter and big dataObjects with big kernel sizes
* OpenCVFilters: filter *cvCvtColor* to color conversion (using OpenCV's method cvtColor), filter *cvCannyEdge* added, filter *cvProjectPoints* added
* DataObjectArithmetic: filter *medianValue* added which is much faster than np.nanmedian due to the quick median implementation
* Roughness: filter plugin for line-wise roughness calculation added (Ra, Rq, Rsk, Rz...)
* SuperlumBS: changed to dataIO plugin type
* DataObjectIO: filter *loadZygoMetroPro* added to load metro pro files from Zygo interferometers
* DataObjectIO: SDF file format can now also be read as binary data
* DataObjectIO: import filter for Avantes files (spectrometers) added
* DataObjectIO: filter *loadNanoscopeIII* added to load Bruker (Veeko) AFM data files
* AvantesAvaSpec: many fixes to operate with many different devices from Avantes (improved handling of dark pixel correction, output with or without correction in double or original integer precision, timestamps added...)
* dispWindow: display of *dataObjects* allowed. If number of phaseshifts are set, the period is adjusted to the next possible value.
* FFTWfilters: filters *fftshift* and *ifftshift* added. They can now also operate along the y- or x- axis only (or both axes)
* FringeProj: filters *calcPhaseMap4* and *calcPhaseMapN* corrected to show the same result for N = 4 (same shape and start-phase).
* DummyMotor: redesign to used MotorAxisController in its dock widget
* CyUSB Plugin added for USB communication via Cypress USB interface
* PIPiezoCtrl: plugin bugfix: If checkFlags are set to 0, the current Pos of the actuator was 0 after a move. m_currentPos[0] will not be set to 0, now.
* ThorlabsISM plugin based on Kinesis 1.7.0 added (Thorlabs Integrated Stepper Motor)
* ThorlabsBP plugin based on Kinesis 1.7.0 added (Thorlabs brushless piezo). This plugin has some known bugs that are described in the documentation due to errors in Kinesis.
* Build plugins *dispWindow* and *glDisplay* only if OpenGL is available (not the case for older version of Raspberry). Only OpenGL >= 2.0 allowed.
* Many filters have been adapted to the new shape class of itom (see :py:ref:`itom.shape` )


Designer Plugins
******************

**Version 2.2.0 (2016-10-01)**

(more than 200 commits in designerPlugins repository)

* itom1dqwtplot: set legendFont also if curve name is changed later
* itom1dqwtplot: added legendFont property
* itom1dqwtplot: re-calculate the sizeHint if the interval limits of the axes change
* itom1dqwtplot: legend of properties widget are connected to the properties dock
* itom1dqwtplot: parent rescale option is only enabled for line cuts, not for z-slices.
* itom1dqwtplot: property antiAliased added to choose if lines should be plot in anti-aliased mode
* itom1dqwtplot: all public properties of a curve are now available over the curve properties dialog
* itom1dqwtplot: remove picker text if no pickers exist
* itom1dqwtplot: markers are kept on the canvas if the bounds of the source object change in values but not in their dimensions.
* itom1dQwtPlot: curve properties 'visible' and 'legendVisible' added. Legend entries show now the line style as well as the symbol of the line instead of filled squares.
* itom1dQwtPlot: default line colors slightly changed to 12-class Paired list from colorbrewer2.org
* itom1dQwtPlot: curve properties can now be changed individually for each curve (slots getCurveProperty and setCurveProperty)
* itom1dqwtplot: legend adjustable via menu
* itom1dQwtPlot: grid of plot can now be adjusted with more options.
* itom1dQwtPlot: added object details
* itom1dqwtplot: icon to "set picker to min/max" added, delete pickers added to toolbar and menu, dx and dy of pickers renamed to width and height (clearer for many users)
* itom1DQwtPlot: improvements for pickers (slots setPicker, appendPicker, deletePicker unified and enhanced; new pickers are also placed to the closest curve, property picker also returns the curve index...)
* itom1DQwtPlot: pickerType can now be AxisRangeMarker (same as RangeMarker) or ValueRangeMarker to provide vertical or horizontal lines
* itom1DQwtPlot: adapted widget to work with new seperation
* itom1dqwtplot, itom2dqwtplot: mouse wheel based magnification with Ctrl, Ctrl+Shift, Ctrl+Alt now also works if y-axis is inverted
* itom1dqwtplot, itom2dqwtplot: plots can be saved via script using the slot 'savePlot'
* itom1dqwtplot, itom2dqwtplot: bugfix
* itom1dqwtplot, itom2dqwtplot: if shapes can be added by the toolbar, the selected shape can also be removed by the Delete-key
* itom1dqwtplot, itom2dqwtplot: geometric shapes can be filled with the line color and a user-defined opacity value. This can be different for the currently selected item.
* itom1dqwtplot, itom2dqwtplot: improved documentations of properties, slots and signals. Slots, that are not accessible via python, are now marked with the ITOM_PYNOTACCESSIBLE macro
* itom1dqwtplot, itom2dqwtplot: further improvements of geometricShapes. Signal 'geometricShapeStartUserInput' added that can be used to disable some buttons during any interactive process of adding new shapes
* itom1dqwtplot, itom2dqwtplot: signal geometricShapeCurrentChanged added
* itom1dqwtplot, itom2dqwtplot: unification and clear calls of signals for adding, changing or deleting geometric shapes
* itom1dqwtplot, itom2dqwtplot: if Shift or Alt is pressed together with Ctrl, only the x-axis or the y-axis is magnified upon a mouse wheel event. If a fixed aspect ratio is set, both axes are magnified in all cases.
* itom1dqwtplot, itom2dqwtplot: created Private container class for all header files of itom1dqwtplot and itom2dqwtplot that are published in the itom designer subfolder in order to better maintain these classes without interrupting the binary compatibility in the future.
* itom1dqwtplot, itom2dqwtplot: property 'enableBoxFrame' added, in order to switch between a boxed and non-boxed frame style of the canvas.
* itom1dqwtplot, itom2dqwtplot: property allowedGeometricShapes added to adjust which type of shapes can be used in this plot.
* itom1dqwtplot, itom2dqwtplot: slots 'modifyGeometricShape' and 'addGeometricShape' added. Shape indices start with 0 per default to have a default corresponding numbering between position in geometricShapes vector and shape index.
* itom1dqwtplot, itom2dqwtplot: print / print preview added
* itom1dqwtplot, itom2dqwtplot: menu restructuring
* itom1dqwtplot, itom2dqwtplot: hint added to interval settings dialogs 
* itom1dQwtPlot, itom2dQwtPlot: left-bottom corner point of y-left and x-bottom axis is connected
* itom1dQwtPlot, itom2dQwtPlot: squares and circles can now also be created by the gui in qwt based plots
* itom1dQwtPlot, itom2dQwtPlot: added functionality of markerInfoWidget and shapeInfoWidget, added PickerWidget to 1D-Plot
* itom1dQwtPlot, itom2dQwtPlot: improvements in moving and resizing geometric shapes
* itom1dQwtPlot, itom2dQwtPlot: ito::Shape used for geometric shapes and picked points
* itom1dQwtPlot, itom2dQwtPlot: deleteMarker slot can also be called without set-name to delete all markers
* itom1dQwtPlot, itom2dQwtPlot: definition of plotMarkers reset to old standard: 2xN, float32 data object where each column represents the (X,Y) coordinate of a marker
* itom1dQwtPlot, itom2dQwtPlot: icons and tooltips improved
* itom1dQwtPlot, itom2dQwtPlot: complete rework of all qwt based plots using unified base classes covering all markers, pickers, geometric shapes...
* itom1dQwtPlot, itom2dQwtPlot: including markerlegend to 2DQwtPlot and 1DQwtPlot
* itom1dQwtPlot, itom2dQwtPlot: changed geometricShapesFinished signal
* itom1dQwtPlot, itom2dQwtPlot: settings of plots are not only read from ini-setting file but are also tried to be inherited from parent plot (e.g. 1D plot as child of 2D plot)
* itom1dQwtPlot, itom2dQwtPlot: added init for markerLegend and changed primitv definition
* itom1dQwtPlot, itom2dQwtPlot: rework of evaluteGeometrics plugin with respect to the new ito::Shape (python: itom.shape) class
* itom1dQwtPlot, itom2dQwtPlot: readded code for markerLegend
* itom1dQwtPlot, itom2dQwtPlot: integrated Qwt fix r2509 from https://sourceforge.net/p/qwt/code/2509/
* itom1dQwtPlot, itom2dQwtPlot: further documentation improvements in itomQwtDObjFigure
* itom1dQwtPlot, itom2dQwtPlot: qwt updated to 6.1.3
* itom1dQwtPlot, itom2dQwtPlot: adaptions due to new library itomCommonPlotLib
* itom2dqwtplot: added signal to 2dQwtPlot to indicate a change of the displayed planeIndex for 3D dataObjects
* itom2dqwtplot: if a line cut or z-stack cut is displayed, the focus is kept in the 2d canvas to receive key- or mouse-events.
* itom2dqwtplot: improved version of valuePicker2d
* itom2dqwtplot: protect line plot to be displayed out of the screen of the primary 2d plot
* itom2dqwtplot: re-position 1D linecut relative to parent figure such that both are not displayed at the same position
* itom2dqwtplot: bugfix in "displayed"
* itom2dqwtplot: value tracker of 2d plot shows pixel and physical coordinates if scaling or offset of dataObject are different than 1 or 0 respectively
* itom2dqwtplot: fix when moving z-stack marker by key-press: the coordinates in the toolbar are actualized now
* itom2dqwtplot: improvements in value picker 2D
* itom2dqwtplot: ignore a keypress event in mode 'linecut' if no line has been drawn yet.
* itom2dqwtplot: working on source / channel updating and changes
* itom2dqwtplot: added function to set a static zSlice to the GUI
* itom2dqwtplot: adapted Plot to work with enum for staticLinePlotState
* vtk3dVisualizer: pyramid can now have a surface.
* vtk3dVisualizer: automatic check for enum pcl::visualization::LookUpTableRepresentationProperties in common.h of PCL (1.8.0). If not available, some features are not available.
* Vtk3dVisualizer: point cloud can now have various color maps including an optional min/max range definition (default: auto-range)
* vtk3dVisualizer: added 'viridis' as color map for polygonal meshes
* vtk3dVisualizer: corrections in CMakeLists.txt
* vtk3dVisualizer: removed fatal error for not building vtk3dVisualizer if itom is built without PCL support
* matplotlibPlot: toolbar and menu is now more consistent with respect to itom1dqwtplot and itom2dqwtplot
* matplotlibPlot: can be forced to replot (slot: replot)
* matplotlibPlot: docstrings added to matplotlibPlot
* matplotlibPlot: improvements: ongoing resize will only be handled at the end or longer pauses of the resize; the paintResult slot can now be called faster via the python based matplotlib backend, since no deep copy of the buffer has to be done now.
* matplotlibPlot: property 'keepSizeFixed' added in order to disallow a automatic resize of the canvas upon a resize of the overall window. This property allow a precise adjustment of the canvas size by the python code (see demo in demo/matplotlib folder of itom)
* others: added available colorMaps in C++ documentation
* others: CMake status messages inserted at begin of every plugin block, itomIsoGLFigurePlugin will not be build if OpenGL is not available (Raspberry Pi)
* others: changes since isNotZero, isFinite... moved from ito::dObjHelper (dataObjectFuncs.h in dataObject library) to namespace ito (numeric.h in common library)
* others: improvements with color and style management. property canvasColor added. see demoPlotStyleSheet.py
* others: used function do remove added Toolboxes from AbstractFigure symmetrically to construction
* others: improving dialog in evaluateGeometrics
* others: adaptions to evaluateGeometrics plugin due to new shape class
* many bugfixes