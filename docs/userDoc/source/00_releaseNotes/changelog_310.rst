.. include:: ../include/global.inc

.. |mm| unicode:: U+00B5 m

Changelog Version 3.1.0
#########################

itom
********


**Version 3.1.0 (2018-03-05)**

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
* fix when opening an info page in the helpTreeDockWidget while the help ist currently rendered or loaded from the database.
* backend_itom.py backward compatible for matplotlib < 2.1
* Added a paletteEditor to create user definied color palettes
* improvements in script reference (help viewer) including PythonAPI_SqlCreator.py
* backend_itom.py: added wait Cursor for Matplotlib 2.1.0...
* fix in copy constructor of DataObject if transpose is True
* added instruction to copy OpenSSL ddl-files into the QT/bin folder for the all-in-one-development-setup
* improvements when typing filter name in helpTreeDockWidget
* fix in pointCloud.fromXYZ, .fromXYZI and .fromXYZRGBA if type of input dataObjects is not the same for all input objects.
* changed parameter checking in AddInOrganizer, as some plugins did not load anymore, when the number of mandatory parameters of the interface was different from the plugin ones
* improvements in PythonAPI_SqlCreator.py
* fix with user rights for property-dialog, unconsistent assignment for user-rights concerning property-dialog
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
* updated pathes in doxygen input file, updates URLs in intersphinx_mappings
* close-method of plugins is only called as direct connection from AddInManager if it is used by an external-DLL and has created its own instance of QApplication. In all other cases, an Auto-connection is used.
* workaround in matplotlib backend if coordinates with special characters (e.g. due to polar plot -> zoom tool, greek letters) should be displayed
* AddInManager unittest: scan plugin directory for plugins instead of entire itom directory
* dataIO.getParam, actuator.getParam: initialize ito::Param value with right type, since this might sometimes be forgotten within the plugin itself.
* ParamEditorWidget: int-array, double-array and char-array: values are displayed and can be manipulated, if desired

Plugins
******************

**Version 3.1.0 (2018-02-20)**

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
* Vistek: sychronization and threading fixes. ROI implemented. For fast acquisitions, plan to insert a small delay before acquiring the next image (this seems to be a problem of the software trigger)
* Ximea: adapt timeout param to integration time (if timeout < integration_time)
* Ximea: added API version switch in CMake (3.16, 4.10, newer)
* Ximea: bug fix Ximea: Serial number of cam is now readable with new api
* Ximea: if camera returns "not supported" error, plugin does not crash, parameter is set to read only
* Ximea: if XI_PRM_DRV_VERSION not supported, serial_number parameter should be empty
* Ximea: make linux plugin ready for api 4.15
* Ximea: Ximea 4.14 API runs with itom plugin



Designer Plugins
******************

**Version 3.1.0 (2018-02-20)**

(more than 120 commits in designerPlugins repsository)

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