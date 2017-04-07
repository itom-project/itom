.. include:: ../include/global.inc

.. |mm| unicode:: U+00B5 m

Changelog Version 3.0.0
#########################

itom
********

**Version 3.0.0 (2017-04-07)**

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
* added a paramEditorWidget which can for eg. can be used to acess parameters of a geniCam instance
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
* added dataobject contructor for single cv::Mat
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
* implemeted a fontButton
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
******************

**Version 3.0.0 (2017-04-07)**

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
* added basci ascii export filter
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
******************

**Version 3.0.0 (2017-04-07)**

(more than 30 commits in designerPlugins repsository)

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
* itom2dqwtplot: added api settings to individualize the z stack picker... added api settings to individualize the label of a drawed item... renamed some api functions
* itom1dqwtplot, itom2dqwtplot: fixes when copying to clipboard or saving to a file (dpi related canvas size corrected). Screen dependent dpi has to be added in the future.
* itom1dqwtplot, itom2dqwtplot: meta information widgets are copied to clipboard if visible
* itom1dqwtplot, itom2dqwtplot: improved styles of picker, shapes, zoomer...
* itom1dqwtplot, itom2dqwtplot: moved all tracker and shape labels by default on a white background. The Color of the letters are no longer changed by changing the color palette of a plot. The geometric element style is now accessible over the api
* itom1dqwtplot, itom2dqwtplot: better acceptance for 2D dataObjects with more than 2 dimensions (e.g. [1,3,3]) in 'plotMarkers' slot
* itom1dqwtplot, itom2dqwtplot: added property complexStyle to set whether the real, imag, abs, or phase in a 1D or 2D plot is displayed
* itom1dqwtplot, itom2dqwtplot: fix to reset cursor in some state transitions (e.g. from panner to line-cut)
* many bugfixes under Linux and Windows