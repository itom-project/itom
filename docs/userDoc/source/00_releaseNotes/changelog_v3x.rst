.. include:: ../include/global.inc

.. |mm| unicode:: U+00B5 m

Changelog Versions 3.x
######################

Version 3.2.1 (2019-06-12)
**************************

itom
----

* install / upgrade dialog of python package manager can now store the current settings during one itom session. This simplifies the installation of several packages.
* due to upcoming support for OpenCV 4, a check for the deprecated CV_USRTYPE1 type of cv::DataType has been removed. This check was responsible to reject the unsupported datatype uint32. Now this check has been implemented again in the create methods for ito::DataObject.
* improvements of std::cout, std::cerr (as well as python print commands) to avoid deadlocks or crashes when printing thousands of lines within a very short time: inserted small delays to avoid buffer overflows.
* menu action "no available figures" set to disabled (like other similar actions)
* documentation added about how to build *itom* under CentOS / linux added


Plugins
-------

* MSMediaFoundation: major improvements concerning necessary CPU consumption (tiny sleeps inserted in while(1) loops)
* GenICam: there exists devices which cannot report the real access state. Instead they report changed the accessStatus DEVICE_ACCESS_STATUS_UNKNOWN. If this is the case, the plugin assumes a read/write access state and tries to open this device though.
* GenICam: Start to support color cameras with the exemplary YCbCr422_8 encoding (tested with Basler puA1280-54uc)
* FittingFilters: small bug-fix in method **fillInvalidAreas**

Designer Plugins
----------------

* itom2dqwtplot: improved positioning of child figures if their preferred position exceeds the geometry of the screen where the parent plot is located.



Version 3.2.0 (2019-04-07)
**************************

itom
----

(more than 440 commits in itom repository)

* increased interface from 3.2.0 to 3.3.0
* complete rework of the script editor and the console. The previously used 3rd party component QScintilla has been removed and replaced by a
  new code editor component. This component is inspired by and ported from the Python editor project **PyQode** (https://github.com/pyQode/pyQode, MIT license).
  It has the following main features:

    * syntax highlighting of Python code
    * code folding
    * auto indentation
    * dynamic auto completion and calltips (requires the optional Python package **jedi**) with the ability to also parse user-defined Python scripts or other packages and
      modules
    * goto definition and assignment of any python method, function, variable or class (also requires the optional Python package **jedi**)
    * on the fly code checkers (requires the optional Python packages **pyflakes** or **frosted** [legacy])
    * most features of the previous editor component have been preserved

* new matplotlib backend (*itom-packages/mpl_itom*): The backend is now compatible with Matplotlib 1.x (legacy), 2.x and 3.x. The user can now change the properties of axes, lines or images via a specific setting dialog.
* improved dark style sheet (*itom/styles/stylesheets/darkStyle*)
* user management: improved documentation, bugfixes and improvements of the user management dialog and better classification of itom features with respect to the appropriate user roles.
* improved documentation, especially the sections about the code editor, the property dialog as well as the user management
* wrap more public methods of important Qt widgets, such that they are accessible by the :py:meth:`itom.uiItem.call` method. See: :ref:`qtdesigner-slots`.
* added or changed demo scripts to the **demo** folder (e.g. *demoDummyMotor.py*, *cloudDemo.py*, *demoContourLines2dPlot.py*, *listWidgetDemo.py*)
* :py:meth:`itom.uiItem.connect`, :py:meth:`itom.dataIO.connect`, :py:meth:`itom.actuator.connect` have an additional optional argument 'minRepeatInterval'. If given (in ms), a python slot is not called more often than this timeout interval. All intermediate calls are blocked.
* point cloud library: replaced deprecated pcl_isnan, pcl_isfinite by std::isnan / std::isfinite
* auto column-size adjustment of docked help widget has been removed (fixes `issue #77 <https://bitbucket.org/itom/itom/issues/77/>`_)
* attributes :py:attr:`~itom.actuator.currentPositions` and :py:attr:`~itom.actuator.targetPositions` added (read-only): They can be used to always obtain the latest reported axes positions of an actuator, even if the actuator plugin is currently executing any method in its own thread.
* avoid deadlock between GUI and python if many plots are closed via the menu or :py:meth:`itom.close`
* dependency check added to pipManager; update button is always enabled, independent if an update is pretended to be available or not
* dialogIconBrowser: bigger icon size for high dpi screens
* package manager and about dialog can be maximized; context menu of table in package manager added to export installed packages to csv file or clipboard
* adapted and improved itom about dialog and license information
* limit the increased recursionlimit (set by jedi) to a lower limit (1600), in order to avoid itom crashes (MSVC built), for example if used together with pandas package.
* improvements of message, shown if a displayed script was externally changed or removed. Distinguish between both cases.
* added new methods :py:meth:`itom.dataIO.connect` and :py:meth:`itom.actuator.connect`, to connect plugin's signals to python methods (like the existing method :py:meth:`itom.uiItem.connect`).
* also added the new methods :py:meth:`itom.dataIO.disconnect` and :py:meth:`itom.actuator.disconnect`
* With respect to the method :py:meth:`itom.uiItem.info`, also added the methods :py:meth:`itom.dataIO.info` and :py:meth:`itom.actuator.info`.
* linux bugfix: workaround for bug in CentOS when waiting for semaphore without timeout value.
* added :py:meth:`itom.dataIO.getExecFuncsList` and :py:meth:`itom.actuator.getExecFuncsList` to print a list of available **exec-functions** of a specific plugin instance to the console.
* added first version of chinese translation
* proper display of *long long* values in workspace
* pip settings of dialog pip manager are now saved and restored to or from settings file
* for Qt >= 5.12, **qcollectiongenerator** has been merged into the **qhelpgenerator**, which now accepts both .qhp and .qhcp files as input and generates .qch and .qhc files, respectively.
* make python calls more robust by also catching all exceptions. This is mainly good to also try to catch stack overflow exceptions (not always possible)
* workspace can now keep track of attributes of a type object which are created by a **__slots__** definition
* editor styles property page: default folder of import and export dialog is the **styles/editorThemes** folder
* fixes some memory leaks, induced by non-decremented PyObject objects
* update of *autoreload.py* (based on https://github.com/ipython/ipython/blob/master/IPython/extensions/autoreload.py).
* info messages are shown if syntax check, auto completion, calltips or 'goto definition' is enabled in properties but pyflakes, frosted+pies or jedi+parso are not available
* :py:meth:`itom.dataObject.dstack` (ito::DataObject::dstack) always returns a dataObject with a higher number of dimensions. (`issue #79 <https://bitbucket.org/itom/itom/issues/79/>`_)
* performance improvements in console widget, if a lot of text should be printed: added an internal intermediate buffer, wrap long lines (optional) into several lines, each starting with ``...``.
* added actuator error and endswitch[no] flags
* python syntax highlighting is now disabled for stream (out and err) output in command line. There is a special style type for these stream outputs (see editor style properties)
* *ito::actuator::getStatus(int axis)* introduced to obtain the status of one single axis
* The python package **pyflakes** is now supported, together with the depcreated package **frosted** to provide syntax checks of scripts.
* dialogReplace: fix layout for high dpi screens
* Added attributes :py:attr:`itom.dataObject.real` and :py:attr:`itom.dataObject.imag` to get and set real or imaginary part of complex dataObject. The former methods *dataObject.real()* and *dataObject.imag()* have been removed in favour of the new attributes.
* scriptDockWidget: avoid unwanted regression in closeAllScripts
* global exception handling: QEvent type name is printed to message (as string, not only number)
* CMake **BUILD_TARGET64** set to default ON
* more source code documentation added
* fixed memory leak in ito::DataObject::stack: planes-stack was not correctly deleted. Further improvements in documentation and some code cleanup in this function
* create designer folder (if it does not exist yet) when building itomWidgets library
* linux bugfix of double-linked-list error when closing itom. The problem was, that the libitomWidgets.so library was copied to both the main directory of itom as well as the designer subfolder. The first version is necessary to run itom, the 2nd version to load ui's containing widgets of this library. However it seems that linux does not like it, if two identical libraries are loaded from two different locations. Therefore, the library in the designer subfolder is now replaced by a symlink to the other library (linux only)
* moved more protected or private members and methods from public interfaces, like AbstractFigure, AddInInterfaceBase, AddInBase,... to avoid many unnecessary ABI changes of the AddInInterface.
* removed **enum.py** from *itom-packages* since it causes compatibility issues with the 'real' **enum.py** of the default python installation.
* fixes related to shutdown of itom: if user cancels the question whether unsaved scripts should be safed, itom is now not closed any more. Further fixes are related to the progress bar in the "close itom" dialog (shown if Python is still running)
* PyEval_InitThreads() has not to be called any more for Python >= 3.7
* Python package manager: support of Pip >= 18.0. Added support for *--trusted-host* options. This allows accessing pypi.org over a proxy in a company network without SSL certificates.
* fix in QSharedPointer deleter method which are responsible to also decrement any PyObject elements. Sometimes, it is not verified which thread (Python, GUI...) is finally releasing a QSharedPointer, which for instance wraps a dataObject, char-buffer... whose base is another PyObject element. Once the pointer is released, a specific deleter-method is called which calls PyDECREF... of the PyObject base pointer. However if PyDECREF is called, the Python interpreter lock (GIL) must be hold by the caller, else a crash might occur. This is now ensured. The solution is only safe for Python >= 3.4, since other versions cannot tell if the GIL is currently hold by the caller or not, which can lead to a deadlock.
* improved error message in global exception catch (QApplication::notify)
* error message if python packages jedi or parso are not available, required for optional auto completion, calltips and goto definition
* PythonDataObject: proper redirection of possible errors in PyDataObj_mappingSetElem
* **clear all** button added in workspace widget of itom, the same feature is also accessible via the new method :py:meth:`itom.clearAll`.
* version tag unifications in all library projects of itom, added missing libraries in version helper of itom
* class :py:class:`itom.shape` has new attribute :py:attr:`~itom.shape.color`. Per default, this color is invalid. In this case, the plots will visualize a shape with the default shape color of the plot. However, if a valid color is set, this color will be used for visualization in plots.
* improvements in widget **pythonLogWidget**: properties *autoScroll* and *verticalSizeHint* added, improved context menu with clear and checkable autoScroll actions
* Plot windows: the optional title of a plot is now also part of the window title, e.g. *Figure 100 - <your plot title>*
* Widget **paramEditorWidget**: property *collapsed* added
* added :py:meth:`itom.dataObject.nans` function for 'float32, 'float64', 'complex64', 'complex128' data type.
* added menu **figure** to itom in order to close, show or activate current figure windows
* dataobj.cpp bug fix for GetStep of dataObject dim > 2
* fix when passing an invalid coordinate tuple to the constructor or static constructors of the class :py:class:`~itom.shape`
* bugfix in *ito::dObjHelper::medianValue* for continuous and / or 2d dataObjects or dataObjects with a limited ROI in the last two dimensions.
* callstack widget: changed column order for a better usability
* the script command for a specific plugin initialization can be obtained via drag and drop from the plugin toolbox to the script or console
* many other bugfixes, for instance:

    * fix to avoid a call for QAbstractItemModel::beginRemoveRows if no rows are currently available in the model (avoids 'new' assertion in beginRemoveRows, added with Qt 5.11
    * linux compiler fixes (e.g. `issue #69 <https://bitbucket.org/itom/itom/issues/69/>`_)
    * fix `issue #67 <https://bitbucket.org/itom/itom/issues/67/>`_
    * Memory leak fix in PythonMatlab::PyMatlabSessionObject_run (`issue #74 <https://bitbucket.org/itom/itom/issues/74/>`_)
    * Windows Setup: Redistributible install error if newer version is already installed (`issue #54 <https://bitbucket.org/itom/itom/issues/54/>`_)
    * prevents a python GIL deadlock, if a np.array is converted to a dataObject and the dataObject is passed to a plot. If the plot then is forced by python to apply another dataObject, the old dataObject is deleted and its base-object (the np.array) should then be deleted, too. However, Python is currently blocked such that the base object cannot be deleted, since the GIL is not free. In this case, the deletion is now put into a concurrent worker thread to be executed when Python is ready again! (`issue #75 <https://bitbucket.org/itom/itom/issues/75/>`_)
    * crash if constructor of :py:class:`itom.shape` is called with invalid or missing arguments (`issue #84 <https://bitbucket.org/itom/itom/issues/84/>`_)
    * when a QSharedPointer is build to share memory with a PyObject pointer, like bytearray or bytes, this PyObject pointer has to be decremented once the QSharedPointer is deleted. This is done in a deleter-method. To decref the PyObject* the GIL has to be acquired. This caused a dead-lock in some cases. This is fixed now. (`issue #81 <https://bitbucket.org/itom/itom/issues/81/>`_)

Plugins
-------

(98 commits in plugins repository)

* all plugins: BUILD_TARGET64 default ON in CMake
* AVTVimba: fixes if some parameters are not supported
* BasicFilters: adaptions to support OpenCV 4
* COCamera: added missing include qlibrary.h
* dataobjectarithmetic: adaptions to support OpenCV 4
* dataobjectarithmetic: many docstrings improved
* DataObjectIO: adaptions to support OpenCV 4
* FileGrabber: adaptions to support OpenCV 4
* GenICam: first running version for CoaXPress cameras
* GenICam: further updates to be able to use Vistek cameras
* IDSuEye: check if memory needs to be reallocated
* IDSuEye: implemented burst mode.
* IDSuEye: update development sources 4.91.0
* MSMediaFoundation: adaptions to support OpenCV 4
* Newport2936: new plugin for Newport power meter devices Newport 1931-C, Newport 1936-R and Newport 2936-R
* OpenCVGrabber: adaptions to support OpenCV 4
* PclTools: replace deprecated plc_isfinite, pcl_isnan, pcl_isinf... by std::isfinite, std::isnan, std::isinf...
* PCOCamera: adapted for new SDK.
* ThorlabsKCubePA: new plugin Thorlabs position aligner.
* ThorlabsKCubePA: sum signal can also be acquired if desired
* ThorlabsPowerMeter: adapted Plugin to run with new SDK (1.02)
* ThorlabsPowerMeter: fix to support legacy Thorlabs Power Meter 1.02 as well as Thorlabs Optical Power Meter 1.1
* ThorlabsPowerMeter: syntax fix in case of old Thorlabs API
* Ximea: changed optional keyword for initialization "camera Number" to "cameraNumber" since it couldn't be used with two words for init

Designer Plugins
----------------

(more than 89 commits in designerPlugins repository)

New features: support matplotlib 3.x including new *edit parameters* dialog, volumeCut in itom2dqwtplot, color of shapes can be assigned individually (in script)

* itom1dqwtplot: dataObjectSeriesData: adapted boundingRect for objects of type ito::Rgba32. Autointerval works now also for rbgba32 objects
* itom1dqwtplot: line cuts in pure x or y direction are now displayed with regard to their start and endpoint. Until now the x-vector was always displayed in positive direction.
* itom1dqwtplot: bugfix when applying z picker on a complex dataObject. The 1DPlot wasn't in the same complex mode as the 2d plot
* itom1dqwtplot: Grid type can now be chosen from menu
* itom1dqwtplot: legend titles from previous object are re-used if no new lines are added to the plot
* itom1dqwtplot: reducePoints is not called for X/Y plots, since this might lead to strange artifacts, depending on the current 'zoom' level or pixel size.
* itom1dqwtplot: set property *pickerType* to Default at startup (was uninitialized before!)
* itom2dqwtplot: adapted since m_lineCutType, m_zSliceType, m_zoomCutType and m_volumeCutType is deprecated and replaced by QMap subplotStates.
* itom2dqwtplot: contour lines will only be printed if at least one level is given
* itom2dqwtplot: added new volume cut feature (similar than line cut):

    * this feature is usable to 3d dataObjects only
    * draw a line, which opens a new itom2dqwtplot, where the vertical planar cut along the line in z-direction is plot as image
    * use the 'h' and 'v' hotkey for a x-z or y-z volume cut
    * for non-parallel cuts to the major axes, the fast Bresenham interpolation algorithm is used
    * added new output channel *volumeCutBounds*, to remove bad interferences between the modes 'line cut', 'z-picker' and 'volume cut'

* itom1dqwtplot, itom2dqwtplot: don't create marker item on canvas for marker positions containing NaN values. Nevertheless, these NaN points are displayed in the 'Marker Info' toolbox.
* itom1dqwtplot, itom2dqwtplot: if shape has a color, this color is used as line color, independent on the inverse color of the current color palette (which is usually used if the shape has no explicit color)
* itom1dqwtplot, itom2dqwtplot: modifications with respect to publishing the title of the current plot in the window bar of its outer window.
* itom1dqwtplot, itom2dqwtplot: unused methods setLabels removed
* matplotlibPlot: avoid rendering regression for multi-screens with scaling factors
* matplotlibPlot: changes in matplotlibPlot to support the variable matplotlib v1.x, v2.x and v3.x backend of itom
* matplotlibPlot: new *edit parameters* dialog for live editing properties of axes, lines, ...
* motorController: if an axis has the new status 'actuatorError', the text fields get a red background (same behaviour than 'actuatorInterrupted')
* vtk3dVisualizer: replaced deprecated pcl_isfinite by std::isfinite
* vtk3dVisualizer: auto-update canvas if scaling changed in properties


Version 3.1.0 (2018-03-05)
**************************

itom
----

(more than 270 commits in itom repository)

* increased interface from 3.1.0 to 3.2.0
* added demo plotXY.py to demonstrate how to set xData in a line plot
* improvements in plot handling, e.g. if itom.plot1, itom.plot2... are used and a className is given, the className is checked to be compatible with the overall plot category (1d, 2d,...). Plot parameter xVec renamed to xData to be compatible to global naming.
* Instructions how to add OpenSSL to future Windows setups added to SetupChangeLog.txt. About.html adapted.
* ParamEditorWidget: it is now possible to set this widget into a mode, that changed parameters are not immediately transferred to the plugin but stored in a cache. The user can then read this cache and set the parameters or force the widget to set all cached changes at any desired time. This allows using this widget in configuration dialogs of plugins, too.
* uiItem (QListWidget) -> methods flags(), setFlags(), checkState() and setCheckState() added as slot, callable via itom.uiItem.call
* FindXerces.cmake improved to also support VS2015 or higher
* added methods figure.plot1, figure.plot2, figure.plot25
* improvements in overall try-catch of application
* paramEditorWidget: representation of integers can show hex numbers
* improved GUI of widgetPropHelpDock
* AutoInterval: takes now double instead of floats (interface increased)
* itom.dumpButtonsAndMenus() added to return a dictionary with all user-defined menus and toolbars that are currently registered in itom.
* demo about multi_processing added (worker threads are opened in external command lines)
* added plot1, plot2 and plot25 python function
* updated version of PythonAPI_SqlCreator.py
* compiler fix for VS2017 and GCC in pythonAutoInterval
* demo added for improved creation of shapes, plotting shapes, moving and rotating shapes... static methods for easier construction added to itom.shape class
* improvements and fixes in ito::Shape class. Added new methods to rotate and translate shape (relative to its center). Still some work to do...
* fixes in shape object: the naming of the components of the transform matrix (based on QTransform) are different than usually used. Additionally, some docstrings have been added to clarify this situation. Still more work to do with translation and rotation...
* itom.shape.copy added to create a deep copy of a shape.
* fix in startupApplication if python startup script is relative, but current directory is different than itom application directory. Relative startup scripts will always be loaded relatively to itom application directory.
* itom.setPalette improved and updated in consistency with itom.getPalette
* itom.getPaletteList and itom.getPalette improved (return values slightly changed, however we assume, that these functions were not used very often until now). Further improvements and code cleanup in color palette editor and color palette manager. Color palette can now be exported and imported in color palette editor.
* paramEditorWidget: waitCursor if parameter is currently updated and bugfix (partially only for x86 systems)
* fixes issue 66: matplotlib history buttons are working properly for matplotlib 2.1.0 and fix for matplotlib > 2.1.0
* rotation angle is displayed in shape info toolbox of plots
* startup script editors in user management and properties: move up and move down buttons added, relative include with respect to itom application directory added as option
* added possibility to add startup scripts when creating / editing user (to avoid necessary change of user type, when adding startup script to user which by its profile could not do this).
* colorMap 'falseColorIR' is now the exact inverse of 'falseColor'
* PythonAPI_SqlCreator.py / helpTreeDockWidget is now able to include Latex-parsed math images (saved in help databases as binary blobs)
* matplotlib example updated to be compatible with matplotlib 2.1.0
* adapted pythonMatlab to newer numpy definitions (at least <= numpy 1.7)
* partial fixes in matplotlib backend to support matplotlib >= 2.1.0. Fixes issue https://bitbucket.org/itom/designerplugins/issues/14/matplotlibplot-idleevent-removed
* fix when opening an info page in the helpTreeDockWidget while the help is currently rendered or loaded from the database.
* backend_itom.py backward compatible for matplotlib < 2.1
* Added a paletteEditor to create user defined color palettes
* improvements in script reference (help viewer) including PythonAPI_SqlCreator.py
* backend_itom.py: added wait Cursor for Matplotlib 2.1.0...
* fix in copy constructor of DataObject if transpose is True
* added instruction to copy OpenSSL ddl-files into the QT/bin folder for the all-in-one-development-setup
* improvements when typing filter name in helpTreeDockWidget
* fix in pointCloud.fromXYZ, .fromXYZI and .fromXYZRGBA if type of input dataObjects is not the same for all input objects.
* changed parameter checking in AddInOrganizer, as some plugins did not load anymore, when the number of mandatory parameters of the interface was different from the plugin ones
* improvements in PythonAPI_SqlCreator.py
* fix with user rights for property-dialog, inconsistent assignment for user-rights concerning property-dialog
* fix if Qt is compiled without SSL support
* updates in PythonAPI_SqlCreator
* helpTreeDockWidget: Improvements with initial position after start, improvements when loading updated databases from internet resource...
* set column span of rows in breakpoint dockwidget, that contain the script filename, only.
* updated fileDownloader and widgetPropHelpDock to improve the SSL-based access to internet resources
* added ito::shape::unclosed to set if a shape object (currently only valid for polygons) is currently being created and therefore still 'opened'.
* fix for finding python root when pipManager is directly started via command line argument
* added possibility to set password for users
* kabsch algorithm improved by version, that can also consider a scaling between two sets of points
* icons for removing breakpoint(s) improved
* raises the optional input boxes or dialogs that are shown if a file is dropped onto the workspace.
* fixes some errors related to wrong usage of Python method PySequence_Fast_GET_ITEM.
* working on rotation of shapes
* bugfix AddInManager crashes when loading plugin with malformed parameter list
* Change compiler flag for c++ exceptions for /EHsc (cmake default) to /EHa, enabling the catching of machine exceptions, e.g. access violation errors
* changed the default drop behaviour of files in filesystemwidget to moveAction. Copy only if ctrl modifier is pressed (this corresponds to the default explorer behaviour in Windows).
* improved file move and copy operation (drag&drop or copy/paste) in fileSystemDockWidget.
* fixes issue #61: error when setting rectMeta to rangeWidget
* preliminary bugfix for issue #62, bug in dataObject transpose
* changed itom to use an instance of qitomapplication by default, enabling exception handling also in release build.
* fix in dialogReplace to also accept caseSensitive versions of the same string (e.g. test and Test) in the comboboxes.
* FindWindowsSDK.cmake, works with MSVC2017
* documentation adapted to current set of parameters of itom2dqwtplot
* made FindITOM_SDK.cmake consistent with ItomBuildMacros.cmake.
* Shortcut for "Run Selection" changed to F9 (matlab-like). Shortcuts added to context menu of script widget.
* fix in detection of Visual Studio version in CMake. FindITOM_SDK.cmake and ItomBuildMacros.cmake have to be consistent!
* adapted creation of itom.bat and itomd.bat to include also VTK directory and boost directory, such that pcltool plugin can be loaded as well
* fixed typo from last commit, changed behavior of build macro such that, starting from MSVC17 the MSVC_VERSION is used rather than MSVC17
* adapted build macros for MSVC detection to MSVC_VERSION variable, as starting from VS2017 (probably) no longer MSVCXX variables will get set
* shape: method contains added to verify of one or multiple points lie within the given shape.
* alpha values of background colors of plot properties in general plot setting dialog are changeable
* brushCreatorButton allows optionally changing the alpha value of the color (can be set via property, if desired)
* added coding keywords to open tex-files for latex docu creation
* File System sorting properties are saved to the settingsFile and are loaded by starting itom to get the same sorting of the file System again.
* bugfix when trying to plot np.array of unsupported types via gui: - added better error messages to some conversion methods between PyObject* and ParamBase (this can be further extended to more types than only dataObjects)
* fixes in navigation bar of FileSystemDockWidget for network drives, starting with \\server\directory\...
* more checks on cv::Mat when creating dataObject from it, trying to avoid malformed dataObjects
* bugfix in msvc17 batch build files
* preparing some stuff for MSVC 2017
* inserted many buffer-overflow checks in DataObject to avoid things like dataObject([-999999999999999999,2]), which was valid before since the huge negative number was -1 (due to the internal buffer overflow - see Python C-API definitions for this)
* fixes for matplotlib >= 2.0: savefig and copyToClipboard should not resize the window if a higher dpi is given.
* fixed wrong (old) signature in getPluginWidget api function and changed some timeout values
* sort output of uiElement.info() alphabetically before showing in console
* adaptions in itom_setup_win32/64.iss.in (e.g. new library addinmanager.dll added)
* Plugin templates update: place the git tag into about string
* Plugin templates update: place the git tag into about string
* adaptions in itom_setup_win32/64.iss.in (e.g. new library addinmanager.dll added)
* Added a aboutInfo function in Python. This returns the about information of a plugin. The about can be used for any additional information (e.g. git hashes).
* updated German translation
* added tr function
* extended "general plot settings" tab in properties by a spin box for the default resolution when copying plots to the clipboard
* fileUtils added to commonQt library to support loading numbers (little endian, big endian...) from raw byte streams or files
* qDebugStream for std::cout and std::cerr initialized and moved to MainApplication. qDebugStream is then globally accessible by AppManagement. Created new AbstractApiWidget (analog to AbstractFigure) to be derived by any widget in order to get access to itom api functions. Created pythonLogWidget to show python output and error streams in any gui. In the future, it can even by possible to stop outputs to the command line, if desired and redirect all outputs to individual pythonLogWidgets in any user-defined gui.
* fix in matplotlib backend to allow properly stopping animated matplotlib figures if the window is closed
* changes in matplotlib backend to support "copy to clipboard" function of matplotlibPlot >= 2.1.0
* removed pythonMessageDockWidget and added pythonMessageBox to itomWidgets
* added legendLabelWidth to generalPlotSettings
* changes of QStringRef.compare commands to compile itom with QT 5.9.0
* itom.shape docstrings improved
* PCL1.8.0 VS2015 bug workaround added to the all-in-one_development_setup description
* dataObject lineCut function adapted to physical coordinates (double)
* fix for WorkspaceWidget if several indices of a Python list should be deleted at once.
* DataObject: enlarged the dStack function. From now on the stacking direction can be specified.
* added widget for Python Messages
* DataObject: better detection of unsupported type uint32
* added 'copy filename' to context menu of script tab
* file name filter mask in fileSystemDockWidget can be filled with semicolon or space separated name list
* dataObject: added a function called lineCut to take a line cut across a 2D or 3D dataObject
* added subdirs of numpy_utils to iss files
* added Python directory for AllInOne installation
* added files for MSVC2015 compilation
* FindWindowsSDK CMAKE-File adapted for MSVC2015
* FindOpenCV.cmake updated for MSVC14
* added titles to compile batch files to better see the type of compilation (debug, release, x86, x64)
* updated paths in doxygen input file, updates URLs in intersphinx_mappings
* close-method of plugins is only called as direct connection from AddInManager if it is used by an external-DLL and has created its own instance of QApplication. In all other cases, an Auto-connection is used.
* workaround in matplotlib backend if coordinates with special characters (e.g. due to polar plot -> zoom tool, greek letters) should be displayed
* AddInManager unittest: scan plugin directory for plugins instead of entire itom directory
* dataIO.getParam, actuator.getParam: initialize ito::Param value with right type, since this might sometimes be forgotten within the plugin itself.
* ParamEditorWidget: int-array, double-array and char-array: values are displayed and can be manipulated, if desired

Plugins
-------

(140 commits in plugins repository)

* added /LTCG linker flag in windows/release build to remove linker warning
* added loading of variables from itom CMakeCache, for faster configuration of projects
* VTK_DIR excluded in load_cache from itom project because without a path this entry can not be changed in the plugin project
* added all gitVersion.h.in files
* the git tag is now added to the about string of each plugin. This function in disabled by default and can be enabled by setting the cmake variable BUILD_GIT_TAG.
* BasicFilters: labeling filter crashes when label touches image boundaries. Hopefully fixed. Probably we need more testing
* BasicFilters: removed 'duplicate' or ghost results from labeling filter
* DataObjectIO: first draft of import filter for keyence topography images (vk4-format). This filter is not added to the list of filters yet, since it is untested. Final version will follow.
* DataObjectIO: first running version of 'loadKeyenceVk4'
* dslrRemote2: make compile with MSVC17
* DummyMotor: better handling of array-based parameters
* DummyMotor: increased default precision in DummyMotor dockWidget, so we can move in steps of nm
* FFTWFilters: implemented fftshift in axis = 0
* FFTWFilters: implemented new parameter in fftshift to shift 3D dataobject in the axis 0
* FFTWFilters: int instead unsigned int for FFTW_FORWARD or FFTW_BACKWARD
* GenICam: error event checked if timeout occurred in acquire/getVal/copyVal method
* GenICam: further fixes to support local xml parameter files
* GenICam: improved handling of case insensitive xml file names
* GenICam: many improvements, runs with Vistek USB3, mono, cameras and now supports mono12packed data format
* GenICam: support for zipped xml parameter files
* Holography. Increasing max field size for Holography filter
* Holography: more sanity checks on input objects in holography filter
* OpenCVFilters: adapted possible input parameters of cvMedianBlur to cv 3.3
* PCLTools: fix for crashes if pointClouds or polygonMeshes are loaded from filenames containing special characters
* PCLTools: savePolygonMesh: binary or ascii mode can now be chosen and for obj-types the decimal precision
* PGRFlyCapture: changed MSVC version detection in ptgrey plugin to be equal to the version used within itom, avoiding potential problems with MSVC 2017+
* PGRFlyCapture: Implemented grab_mode for PtGrey CameraInfo
* PGRFlyCapture: parameter 'num_idle_grabs_after_param_change' added to optionally grab X dump images after parameter changes in order to get a clean new image with the new properties
* PIHexapodCtrl moved from internals to plugins
* PmdPico: added PMD Plugin: tested with pmd pico flexx cam... requires royale software (header, lib and dll are not shipped with the plugin)
* rawimport: added some exif(tool) information to rawImport filter - not tested on linux, probably needs some attention there
* rawimport: increase waiting time for exifTool
* rawimport: labeling: handling of empty label list, rawimport: changed file extension for dcraw input
* SerialIO: fix in dialog to choose, if a getVal should be directly executed after each setVal. This is not always desired, especially if the readline flag is activated.
* SerialIO: removed restriction to endline characters. From now on any sign can be used...
* ThorlabsPowerMeter: Updates in Thorlabs PowerMeter, a check for PowerMeter Version is integrated
* Vistek: synchronization and threading fixes. ROI implemented. For fast acquisitions, plan to insert a small delay before acquiring the next image (this seems to be a problem of the software trigger)
* Ximea: adapt timeout param to integration time (if timeout < integration_time)
* Ximea: added API version switch in CMake (3.16, 4.10, newer)
* Ximea: bug fix Ximea: Serial number of cam is now readable with new api
* Ximea: if camera returns "not supported" error, plugin does not crash, parameter is set to read only
* Ximea: if XI_PRM_DRV_VERSION not supported, serial_number parameter should be empty
* Ximea: make linux plugin ready for api 4.15
* Ximea: Ximea 4.14 API runs with itom plugin


Designer Plugins
----------------

(more than 120 commits in designerPlugins repository)

* plotxy: setXObj returns from now on a RetVal... This can later be used to return a RetVal from a plot
* itom1dqwtplot, itom2dqwtplot: distance of rotation marker to shape is set to 25px instead of a fixed distance in scale coordinates
* itom1dqwtplot: input parameter xVec renamed to xData to be compatible with itom and the global naming
* itom1dqwtplot: fixes for C++11 incompatible compilers (like VS2010)
* plotxy: added multi column mode
* plotxy: allow one xVector for each row in source dataObject...
* plotxy: setPicker slot adapted, send current view to work space is now disabled for xy-plots
* plotxy: valueUnit and description of xVector dominates over source axisUnit and description
* plotxy: bugfix for picker update when changing complex state
* plotxy: added hash check for x-vector
* itom2dqwtplot: color maps can be also selected from submenu of existing colormap action
* plotxy: do not call drawSeries with coordinates... this leads to lost connections between the points
* itom1dqwtplot, itom2dqwtplot: further fixes with respect to shapes
* plotxy: switching between xy and 1d implemented
* itom1dqwtplot, itom2dqwtplot: improvements for rotating, moving... shapes in plots
* improvements in shape handling (movement, rotation, resize)... based on new methods of ito::shape class. Still work to do... (will follow soon)
* itom2dqwtplot: better coordinate representation in valuePicker2d
* plotxy: first work on switching between xy and normal 1d plot
* itom1dqwtplot, itom2dqwtplot: improvements when rotating shapes by mouse
* z-stack cursor is now always using inverse color of current palette
* itom1dqwtplot, itom2dqwtplot: improvements when rotating shapes by mouse
* z-stack cursor is now always using inverse color of current palette
* itom1dqwtplot: bugfix when setting property 'unitLabelStyle'
* itom2DQwtPlot value description + unit coding bugfix
* plotxy: x-vector is now checked in Plot1DWidget
* VTK_DIR excluded from load_cache because if this entry was not defined by the itom-project, it could not be overwritten in the designPlugins-Project.
* itom2DQwtPlot axis description + unit coding bugfix
* Matplotlibplot: added wait Cursor for Matplotlib 2.1.0... Maybe that there are  some more fixes needed since IdleEvent class is deprecated in version 2.1...
* itom1dqwtplot, itom2dqwtplot: added propertyWidget Button to toolbar
* qwt: implemented fix from upcoming qwt version 6.3
* Polygon shapes, adding and removing of single points
* itom1dqwtplot, itom2dqwtplot: changed from flag 'PolygonOpen' of ito::shape to member unclosed.
* itom2dqwtplot: fix when changing the property 'overlayAlpha' to 0 or 255.
* vtk3dvisualizer: VTK errors are printed to text file 'vtk_errors.txt' instead of being displayed in popup window, which might crash.
* itom1dqwtplot, itom2dqwtplot: improved valuePicker for very big or small floating point values
* started to implement updateDataObject in dataObjectSeriesDataXY.cpp
* crash when scaling image color span on linecut limits and upper bound was a #nan
* 1D-Plot, crash when spawning line on maybe strange dataObject
* added loading of variables from itom CMakeCache, for faster configuration of projects
* itom2dqwtplot: bugfix concerning new dataChannel property + icons added
* itom2dqwtplot: for RGBA32 dataObjects, the displayed dataChannel can be chosen (either RGBA as color, or single color channels or gray value from coloured object)
* fixes issue 12: 2D plot and z-cursor are now working for negative axisScale values
* itom1dqwtplot, itom2dqwtplot: fixes issue 11. if plot is in an ui, properties that can be set in QtDesigner will not be overwritten by global itom settings
* itom1dqwtplot: improved placement of new pickers, since they snap to the closest point on a line to the current cursor position within a small x-range. If a picker is active and the key-buttons are pressed, the mouse-cross does not move any more.
* itom1dqwtplot: set picker at local min/max position within current view implemented (fixes issue #13)
* DataObjectSeriesData: removed template from header by using explicit template instantiation
* send current view to Workspace exports complex datatype from 1d plot. It was necessary to realize this template function in the header file.
* itom1dqwtplot: improvements in picker-handling. If multiple curves are visible, pickers can only switch between visible curves using the up and down buttons. If the min-max-action is clicked and multi curves are visible, a selection dialog appears to ask the user to select one (visible) curve.
* itom2dqwtplot: fix when applying a line cut with Ctrl-modifier and dataObjects where the size of the x- and y-axis vary very much. In the new version, the right quadrant is checked based on the screen coordinates instead of the scale coordinates.
* fixed crash of 1D plot, when spawning line on awkward data object
* itom1dqwtplot: fixes parameterizations for different curve styles, such that baseline is properly displayed if desired.
* itom1dqwtplot, itom2dqwtplot: fixes issue #10 (crash when exporting canvas to clipboard if any toolbox is visible and undocked and if a shape is highlighted)
* itom1dqwtplot, itom2dqwtplot: improved text fields in scale-setting dialogs (e.g.scientific notation possible for min/max values of axes)
* let Itom1DQwtPlot::getLegendLabelWidth() return a value in all cases
* itom 1D plot adapted to export current to workspace with complex datatypes. In case of a linecut of 2D plot, it does not work!
* itom1dqwtplot, itom2dqwtplot: "Current to workspace" adapted for complex datatypes. User can choose by a checkBox if the complex datatype should be send to workspace
* itom1dqwtplot, itom2dqwtplot, matplotlibPlot: the dpi of the "copy to clipboard" method can now be set by one common default value (in the itom property dialog)
* matplotlibPlot: copy to clipboard (Ctrl+C) implemented with default dpi of 200 (similar to itom1dqwtplot and itom2dqwtplot)
* itom1dqwtplot, itom2dqwtplot: changed default dpi for "copy to clipboard" to 200 dpi (due to possible import errors in MS Office... if the image is too big). The default dpi should be changeable by the itom figure properties in future releases of itom.
* Itom2dQwtPlot: bugfix styles will no longer be set to default if the color palette changes
* itom1dqwtplot, itom2dqwtplot: fix in wheel-based scrolling if one or both axes are logarithmic
* 1dQwtPlot: added property legendLabelWidth
* itom2dqwtplot: fix bug if plane > 0 is selected and the complex type is changed, the plane was implicitly reset to the first plane. Now the plane is kept the same.
* itom1dqwtplot: 12th default line color slightly changed to become a little bit darker
* itom1dqwtplot: faster plot of very huge dataObjects (e.g. > 1e5 or 1e6 points) in case of ordinary line plots (no steps etc.)
* itom1dqwtplot: data points outside of current view are not considered any more for data handling (if ClipPolygons attribute is On). This increases plotting speed for big dataObjects and zoomed views.



Version 3.0.0 (2017-04-07)
**************************

itom
----

(more than 250 commits in itom repository)

* demoDataObject.py modified for a clearer overview of important features
* demo about image distortion simulation and determination added
* demo for paramEditorWidget added to demo/ui
* fixes due to high-dpi display
* added message in the CMake-files in pluginTemplates to print the plugin name in the CMake output
* documentation: workaround of vtk linker exception under linux
* added some classinfo descriptions for motorAxisController and paramEditorWidget
* added documentation with demo script of how to set legendTitles in a 1d plot with the help of tags.
* update documentation of 1d and 2d plot
* updated documentation due to modified widget wrappers (QComboBox::itemText accessible via python call)
* fix in ItomBuildMacros.cmake, macro POST_BUILD_COPY_FILE_TO_LIB_FOLDER if the given library list is empty
* added a paramEditorWidget which can for eg. can be used to access parameters of a geniCam instance
* added itemText method to qlistview (widgetWrapper) and some code formatting
* documentation updated for new pickerChanged signal of itom1dqwtplot
* updated setup documentation of itom
* added python_unittest function checking the behavior of dstack
* modified widgetWrapper member function to return ito::RetVal instead of bool, for better error diagnostics
* style sheet adaptation for plugin documentation
* fixes for some missing type registrations in Qt meta type system
* added python-based cameraToolbox, containing a module 'distortion' to determine radial distortion, scaling and rotation of a distorted grid of points (including their undistortion). This can be used for characterizing optical systems e.g. based on single chessboard or regular point-grid patterns.
* incremented AddInInterface to 3.1.0 due to further, binary-compatibility breaking changes in AddInManager and other parts
* from Qt5.5 on, an automatic parsing of enumerations and their keys in terms of strings is possible. Added this feature to improve python slot-calls.
* added a static dataObject function called dstack to stack different dataObjects
* added a new python command to copy the whole dataobject meta information of a dataObject
* pythonEngine, load Variables: name/key compatibility improvements
* itom_unittests dataObject saturate_cast_test fix problems with rounding of rgba to gray values.
* itom_unittests operatorTest changed, that the operation: matrix divided my matrix has no zero values.
* bugfix in ito::DataObject::deepCopyPartial for 1xm vs. mx1 objects where the lhs object is a roi of a bigger object. Including unittest in python_unittests environment.
* added marker for docked or undocked scriptEditorWidget
* improvements in unittest: application will be build in itom-build directory for a better access to required libraries
* show size of numpy array in detail window
* unittest and changes in dataObject::copyTo (no reallocation if rhs dataObject has already the right size and type)
* changed behavior of copyTo, rhs is not reallocated when regionOnly is passed and it already has the correct type and size.
* improvements in unittest: application will be build in itom-build directory for a better access to required libraries
* added unittest for externally started AddInManager
* Qt4 compatible version of AddInManagerDLLr
* updated German translation
* some security checks and comments added to new constructor of DataObject.
* changes in tex_demo.py. There seems to be a bug in matplotlib 2.0.0, when using savefig command in combination with unicode!?
* added dataobject constructor for single cv::Mat
* added some documentation about the Latex support for Matplotlib Text layout
* re-add missing LinguistTools library
* Integrated changes in commontk project: https://github.com/commontk/CTK/commit/5db3f9320ed50b9d8179236cd3e84694dd7153ec
* Updates in conf.py of user documentation due to changed default settings in QtHelpBuilder of Sphinx
* update all-in-one development setup documentation
* documentation: brief section of the global plot settings. Added some new settings
* improved documentation about unfound font in Matplotlib
* added max function for complex valued dataObjects, returning the largest magnitude (and its position)
* Added function to check for gui support to addInInterfaceBase and AddInBase
* added more features to some ParamMeta classes
* contributors adapted
* removed some memory leaks in Python C-code and removed some unused methods in PythonEngine
* fixes issues with special characters in filename of itom.loadMatlabMat and itom.saveMatlabMat
* some test scripts with scipy added (can also be used to check proper installation of scipy)
* added Redistributable Visual Studio 2015 to setup
* documentation: package list for linux updated
* more cmake information for linux
* added default style settings for qwt plots into the properties
* redesign of dataObjectFuncs to provide a clearer helper class for verifying, comparing and preparing dataObjects
* domain itom.bitbucket.org replaced by itom.bitbucket.io
* ito::ParamBase::Type: flag NotAvailable added for temporarily unavailable parameters
* added PenCreator Button, BrushCreatorButton and FontButton to UI-Designer
* ito::Param::getMetaT<_Tp>() added as templated-version for ito::Param::getMeta() such that the C-cast can easily be replaced by the template parameter.
* Bugfix application stall on closing, when using AddInManager dll
* overloaded sizeHint in PenCreatorButton and added a brushCreatorButton
* implemented a fontButton
* added "item" slot for qListWidget, which retrieves the text of the "item"th row
* added version.h (qitom / git) to sdk, added apifunction to read out filter version, author and plugin name
* added option for Python path as itom subdirectory
* added commonQtVersion.h and pclVersion.h to SDK files
* AddInManager now starts an instance of QCoreApplication if none is found
* unified names between IOHelper and FileSystemDockWidget
* added some missing metatypes
* added missing pcl integration
* pip manager adapted to changes for pip >= 9.0
* documentation added how to build itom under Fedora (tested with Fedora 25)
* changed order of methods in dataobj.cpp to support gcc under Fedora
* CMake syntax fix for CMake < 3.0.0
* prepare FindOpenCV.cmake to support Fedora
* moved apiFunctions to addin manager and restructured them, i.e. removing gui related functions. These were moved apiFunctionsGraph (which should be renamed to apiFunctionsGui actually)
* added addinmanager to sdk lib files, some minor cmake cleanup
* docstrings of itom.pointCloud, itom.point and itom.polygonMesh improved
* Clean up of unused includes
* plugin widgets can now be opened by plugin dock widget even if they have mandatory or optional initialization parameters
* Move AddInManager to DLL initial version
* property editor has context menu entry to select between sorted and unsorted representation
* pointCloud.moveXYZ added (offsets every point in a point cloud, comparable to pointCloud.scaleXYZ)
* closeItom event can be stopped by click on the Cancel button or press the ESC Key on the keyboard.
* console: double click on any type of warning (not only runtimeWarning) will open the corresponding script.
* make matplotlib itom backend backward compatible with matplotlib 1.4.x
* added openmp to dataObject
* dialog added to close itom, even if python is still running. User will be asked if itom really should be closed.
* itom debug icons changed into 64x64 size for 4k optimization. svg files of the icons added, too.
* itomWidgets: plotInfoDObject can handle rgba32 dataObjects
* improved error messages
* fix in matplotlib backend: avoid using single-shot zero-delay timer in "soft-draw" method to only redraw if the gui is idle. However, an itom.timer cannot be used in python multi-threaded applications
* matplotlib demo about surface3d plot added
* warning added to itom.timer if it is started in another python thread
* docked toolbar icons as well as margins of script editor are scaled based on screen dpi
* set remote search for Python documentation to python version 3.5
* global rework of all icons
* support 4k monitors: changed icons to 64x64 px and scaled UI items by the ratio (screen dpi / default dpi: 96)
* added pyparsing-2.1.10 to setup
* bugfix in WorkspaceDockWidget::updateActions(). Sometimes the return value of m_pWorkspaceWidget->numberOfSelectedItems() is different to the return item.size() of m_pWorkspaceWidget->selectedItems()
* pluginTemplates updated
* many bugfixes under Linux and Windows

Plugins
-------

(nearly 100 commits in plugins repository)

* PGRFlyCapture: parameter 'start_delay' added to possibly allow blue screens when starting the device, followed by an immediate image acquisition
* test string parameters with different meta information added to DummyGrabber in order to better test the ParamEditorWidget and its demo, that is based on the DummyGrabber and DummyMotor.
* NITWidySWIR message, that NUC file was loaded deleted.
* parameter "name" in some plugins set to readonly
* updated plugin rst changelog
* DummyGrabber, DummyMotor: enhanced meta information for parameters to be better rendered with ParamEditorWidget
* fixes in GenICam if camera does not provide enough information about the image buffers
* GenICam plugin (currently only B/W cameras implemented) added. Requires GenICam v3.0 or higher to compile.
* adapted all ui files of plugins for the use of high dpi screens
* NITWidySWIR added message to print the plugin name in the CMake output
* Bugfix in DummyMotor jogging
* CyUSB plugin compiled with version 1.3.3
* MeasurementComputing plugin compiled with SDK Version 1.89
* added NITWidySWIR plugin
* inserted params into documentation of ThorlabsPowerMeter
* ThorlabsBP and ThorlabsISM updated to Kinesis 1.9.3 (from Kinesis 1.8.0 on, the parameters 'slewRate' of brushless piezos have been removed, since they were never implemented in the controller).
* removed wait_for_event from DslrRemote2 capture, as in the current form not working with Nikon Dslr - needs checking
* removed unused opencv calib3d from BasicFilters
* added automatic normalization to labeling filter, as without this the filter does not work correctly
* Moved labeling and findEllipses filter from ito internal plugins to BasicFilters
* BasicFilters: calcMeanZ: max z-depth is not limited
* DslrRemote2: Added execfunc to retrieve file list and to download file by name
* Working on DslrRemote2, first working version (windows), Linux still needs to be checked; Currently reading properties, capturing images and basic version of image download are working
* dataObjectArithmetic: improved mode 4 of filter 'findMultiSpots'
* dataObjectArithmetic: improvements in findMultiSpots
* added filter 'findMultiSpots' to DataObjectArithemtic in order to quickly find multiple spots (using the center of gravity) within one image (or a stack of images).
* DslrRemote, next try ... this time without libgphoto and only with libptp2 (strongly modified version, using new libusb)
* MSMediaFoundation: improvements to support more parameters especially for new HD webcams with full focus and white-balancing control
* rawimport added clean up of temporary directory and detection of dcraw output format (pgm or tiff)
* creation of all dock widgets protected by check if plugins are loaded by GUI-itom (not directly by externally loaded AddInManager)
* OpenCVFilters: corrected docstring of cvCalibrateCamera
* changed CMake to follow changed rules of CMake > 3.0.0 in order to prevent policy warnings
* Vistek: add missing VistekGigESDK files
* division of pclTools ModelFitGeneric file into various files due to linker bogobj issue
* SVS Vistek GigE SDK updated to 1.5.2
* DemoAlgorithms: demoWidget added
* more robust implementation of saveDataToTxt in DataObjectIO (concerning empty wrapSign and linux compilation)
* Config dialog of FireGrabber adapted to modern style. Parameter 'roi' added.
* added basic ascii export filter
* modification in ptgrey path detection
* Added newer MSVS version to PtGrey find library / binary
* DataObjectIO: filter 'loadTXT' adapted and optional encoding of text file added. If this is not given, the encoding of the text file is guessed.
* timeout parameter added to UhlText and UhlRegister
* IDSuEye prepared for SDK 4.81 which has a better support for some modern USB3 cameras
* added optional limit switches and jogging capability to dummyMotor, enabled event processing in waitfordone so asynchrone getpos are possible
* PCOCamera: this plugin can now also operate PCO.edge cameras (tested with PCO.edge 3.1)
* AndorSDK3: delay inserted into stopDevice() to avoid feature access errors
* IDSuEye also works with Thorlabs DCC and DCU cameras. The plugin ThorlabsDCxCam is obsolete now.
* PclTools: fix in loadPolygonMesh and loadPointCloud for ply format. Some docstrings improved
* ThorlabsISM: Character encoding fix
* ThorlabsPowerMeter: docstring fixes
* All work done... ready to use the ThorlabsPowerMeter plugin
* docstring fix in deviationValue of dataobjectarithmetic
* Bugfix in filter histogram; the parameter for definition of bin numbers is now used for 8bit images correctly
* many bugfixes under Linux and Windows

Designer Plugins
----------------

(more than 30 commits in designerPlugins repository)

* others: from Qt5.5 on, an automatic parsing of enumerations and their keys in terms of strings is possible. Added this feature to improve python slot-calls.
* others: changed Icons to 64x64 size.
* others: linux Qt4 bugfixes, thanks to Goran
* others: bugfix in vtk3dVisualizer: yAxisLabel and zAxisLabel showed the same value
* others: corrected format of menu and messages
* others: re-add missing LinguistTools library
* others: changed CMake to follow changed rules of CMake > 3.0.0 in order to prevent policy warnings
* itom1dqwtplot: fix for setting legend fonts if some legends are hidden
* itom1dqwtplot: legendTitles can again be set via dataObject tags
* itom1dqwtplot: added "active picker" to 1DPlot, i.e. active picker can be get/set via the property ...CurrentPickerIndex, in addition the signal pickerChanged when a picker has been altered
* itom1dqwtplot: docstring added for pickerChanged signal
* itom2dqwtplot: fixes in automatic linecuts at horizontal or vertical lines containing the min/max value
* itom2dqwtplot: fix to rescale color bar to right values if the value range is set to 'automatic'.
* itom2dqwtplot: color bar can now display a logarithmic or double-logarithmic scale
* itom2dqwtplot: a zStack or a lineCut is now of the same complex type as the parent plot
* itom2dqwtplot: added api settings to individualize the z stack picker... added api settings to individualize the label of a drawn item... renamed some api functions
* itom1dqwtplot, itom2dqwtplot: fixes when copying to clipboard or saving to a file (dpi related canvas size corrected). Screen dependent dpi has to be added in the future.
* itom1dqwtplot, itom2dqwtplot: meta information widgets are copied to clipboard if visible
* itom1dqwtplot, itom2dqwtplot: improved styles of picker, shapes, zoomer...
* itom1dqwtplot, itom2dqwtplot: moved all tracker and shape labels by default on a white background. The Color of the letters are no longer changed by changing the color palette of a plot. The geometric element style is now accessible over the api
* itom1dqwtplot, itom2dqwtplot: better acceptance for 2D dataObjects with more than 2 dimensions (e.g. [1,3,3]) in 'plotMarkers' slot
* itom1dqwtplot, itom2dqwtplot: added property complexStyle to set whether the real, imag, abs, or phase in a 1D or 2D plot is displayed
* itom1dqwtplot, itom2dqwtplot: fix to reset cursor in some state transitions (e.g. from panner to line-cut)
* many bugfixes under Linux and Windows
