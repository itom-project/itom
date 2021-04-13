.. include:: ../include/global.inc

.. |mm| unicode:: U+00B5 m

Changelog Version 4.0.0
#########################

itom
********

**Version 4.0.0 (2020-05-28)**

(more than 300 commits in itom repository)

**New or changed major features:**

* ScriptEditor can now handle filenames with special characters
* New alternative interface for algorithms (ito::FilterDefExt) in algorithm plugins. This allows both continuously reporting the progress of an algorithm to Python (class py:class:`itom.progressObserver`) or C++ GUIs as well as interrupting a long running algorithm from these GUIs, too.
* The keyword '_observer' is not allowed in any algorithm plugin filters, since it is a reserved keyword for passing an progress observer to a filter. Filters having such a keyword will be rejected by itom.
* Forward and backward navigation buttons in script editors are available. Their behaviour is similar than in Visual Studio (see https://blogs.msdn.microsoft.com/zainnab/2010/03/01/navigate-backward-and-navigate-forward/)
* Improved callstack toolbox: The internal traceback of the debugger is ignored, now. It is possible to double click every level of the callstack in order to jump to these lines. The top level will be marked with a yellow arrow in the script, other affected lines by a green arrow.
* Added an alternative, optional code checked based on the Python package **flake8**. This enhances the functionality of **pyflakes** and displays more information in the first margin of the script editors. The code checker can be heavily configured by the property dialog of itom.
* Bookmarks in all script are now managed by one bookmark model. A toolbox for all recent bookmarks is now available. The bookmark forward and backward buttons will now navigate over all bookmarks in all scripts (`Issue 112 <https://bitbucket.org/itom/itom/issues/112>`_).
* Press F12 if the cursor is within a word in a script to trigger the "Goto definition" action for the word under the cursor.
* The user can now select the keyboard modifiers that are used to start a "goto definition" operation in a script upon moving the mouse cursor over a word in the "goto assignment" property page. Default: Ctrl+Shift.
* Shortcuts to open an existing or new script, as well as shortcuts to debug any script are now globally available (see `Issue 115 <https://bitbucket.org/itom/itom/issues/115>`_).
* The standard user can now also have a password (see `Issue 88 <https://bitbucket.org/itom/itom/issues/88>`_).
* More members of :py:class:`itom.dataObject` now also accept keyword-based arguments
* :class:`itom.ui`: new window type ui.TYPECENTRALWIDGET added. Use this type to permanently include this widget or QMainWindow to the central area of itom, on top of the command line (see :ref:`qtdesigner`).
* Drop of Qt4 support in itom and its plugins.
* More robust compatibility check (by means of Semantic versioning) when loading designer plugins, to avoid crashes if incompatible plugins are loaded (this requires modifications in designer plugins).
* New interface :py:meth:`~itom.dataIO.stop` as well as ito::DataIO::stop() to allow stopping continuous acquisition or write operations for ADDA (I/O) devices.
* Refactoring of all CMake files to follow the new CMake style guide rules of itom (see :ref:`cmake-style-guide`). The CMake files for the itom SDK are now in a **cmake** subfolder. Macros in **ItomBuildMacros.cmake** are now renamed and start all with the prefix *itom*. The minimum CMake version is now 3.1. Many unused preprocessors, useless things etc. removed from CMake files.
* CMake: Plugins will now try to automatically detect the ITOM SDK in some standard directories and read some 3rd party libraries from the CMakeCache.txt file of itom (if possible).
* :py:meth:`uiItem.getChild` (widgetName) added as alternative for **uiItem.<widgetName>**, since the first method can also be used if the widgetName is a variable of type str.
* Widget wrapper 'setItemText' for QListWidget added. gui.listWidget.call("setItemText", 0, "new text") changes the text of the first item in the given list widget.
* The script reference window (help viewer) of itom is now renamed to "plugin help viewer" and only shows information about plugins. The former script reference, based on offline database, hosted at sourceforge.net/p/itom, has been removed since the live script reference (using the python package jedi) fully replaces this technique. OpenSSL is no more needed now.
* Modifications of the license information of itom, add of the new licenses folder in the itom sources with all major 3rd party projects that are used in the core as well as commonly used plugins/designer plugins of itom.
* Python **help** output can now also be opened in external editor. This can be configured in the itom property dialog, page **Python >> General**
* Some more demo scripts added (e.g. face detection via OpenCV, settings the color of shapes, starting the roughness evaluator, or cancelling long running algorithms including showing their progress in own GUIs
* Improved icon browser dialog: icons can be filtered by a text box. The load is put into a concurrent run task to improve the startup. name of selected item is displayed in textbox below the tree widget.
* :py:meth:`itom.pluginLoaded` now only allows the name of a plugin, not the filename of a plugin library.
* Some fixes in python auto indent to provide a better indentation after line breaks based on the pep8 indentation rules
* Selection from auto completion list is only confirmed with the Return key in a script. In the console, only the Tab-Key is feasible.
* New property page for actuators: It can be chosen if an interrupt flag should be send to all active actuator instances if a python script execution is interrupted (default: false). Calling itom.setPosAbs, itom.setPosRel, itom.calib or itom.setOrigin will now reset the interrupt flag before execution (as well as calling these methods from the motorAxisController.
* Macro 'itom_fetch_git_commit_hash' added to ItomBuildMacros.cmake to get the current Git commit hash and store it in a gitVersion.h file in the file gitVersion.h in the project's output folder (can be changed). This behaviour can be toggled by the BUILD_GIT_TAG flag in CMake. This can only be done if the Git package can be found by CMake.
* Script editor tabs: it is now possible to configure how long filenames are shortened if there are many scripts opened (see itom property dialog >> editor >> script editors)
* File system dock widget: list of recent folders, loaded at startup from settings, will only contain pathes that exist at startup.
* Grabber plugins can now have an optional **sizez** parameter and must then return a 3D dataObject (image stack) with shape (sizez, sizey, sizex)
* Added the new default editor style *DefaultConsolas.ini*, that is based on the default style, but uses the consolas font for all style types.
* The type **ItomPlotHandle** can now be set to None in Python. This allows removing assigned plot widgets for line cuts, z-stack cuts etc. and remove this connection between two plots.
* The python package **breathe**, required to build the user documentation, is no longer shipped together with the itom sources, but must be installed as official Python package (for easier updates)
* Python package manager of itom can now also install packages from a **requirements.txt** file.

**Bugfixes:**

* `Issue 87 <https://bitbucket.org/itom/itom/issues/87>`_: multiple files can now be opened by droping on script editor
* `Issue 89 <https://bitbucket.org/itom/itom/issues/89>`_: Removed a wrong "container=1" line in the ui-files for itom designer plugins. This line will let the QtDesigner crash if another widget is drag&dropped over a widget of the affected designer plugin class.
* `Issue 94 <https://bitbucket.org/itom/itom/issues/94>`_: method navigation combobox above script editor did not show methods with some typehints.
* `Issue 95 <https://bitbucket.org/itom/itom/issues/95>`_: correctly highlight private methods in scripts, that have numbers in their method name.
* `Issue 96 <https://bitbucket.org/itom/itom/issues/96>`_: bookmark icons are directly removed if 'clear all bookmarks' is clicked
* `Issue 97 <https://bitbucket.org/itom/itom/issues/97>`_: removed uint32 from docstrings in itom.dataObject (since not supported)
* `Issue 98 <https://bitbucket.org/itom/itom/issues/98>`_: camera can be disconnected from plot by assigning **None** to the camera property
* `Issue 100 <https://bitbucket.org/itom/itom/issues/100>`_: Bugfix when obtaining the variable name from a selected sub-item in the workspace tree.
* `Issue 104 <https://bitbucket.org/itom/itom/issues/104>`_: corrected ascending or descending sorting of elements in the workspaceWidget if values in variable name column are numbers, represented as texts. Therefore "10" should follow "2" instead of the text-based comparison.
* `Issue 106 <https://bitbucket.org/itom/itom/issues/106>`_: drag&drop from/to command line: - it is not allowed to drop something in protected lines of the command line - dragging from protected lines of the command line must always be a copy operation
* `Issue 109 <https://bitbucket.org/itom/itom/issues/109>`_: Commands, added to the recent list of commands, are only considered as duplicates if their command string is equal to any older command in a case-sensitive way.
* `Issue 110 <https://bitbucket.org/itom/itom/issues/110>`_: bugfix in scope decoration of code editor.
* `Issue 113 <https://bitbucket.org/itom/itom/issues/113>`_: Current selection of script removed upon a mouse right click in script.
* `Issue 114 <https://bitbucket.org/itom/itom/issues/114>`_: SystemError when converting an empty np.ndarray to an itom.dataObject (python unittest added to reproduce this error)
* `Issue 120 <https://bitbucket.org/itom/itom/issues/120>`_: bugfix in shape.createPoint, unittest added to verify this bug.
* `Issue 121 <https://bitbucket.org/itom/itom/issues/121>`_: The dialog, displaying the content of a variable in the workspace, will now be displayed as non-modal dialog on top of the workspace widget.
* Matplotlib backend: fixes several bugs in matplotlib backen (e.g. due to deprecated arguments in matplotlib 3.x) 
* Redo button in script editor is now working properly
* Workaround in font selection of WidgetPropEditorStyles
* Bugfix: memory leak for copy constructor **itom.dataObject(cpy: np.ndarray)**
* Bugfix in itomWidgets: ledStatus reported wrong header file, which made it hard to insert it into an ui file
* Reference counter of ito::ByteArray is now incremented or decremented atomically. This improves the usage of a ByteArray within different threads.
* AddInManager: fix when closing plugin instances, opened via GUI, in destructor of AddInManager: instances should only be directly closed by AddInManager if they are not only referenced any more by any other plugin instance(s). In this case closing the owning instance will also close the referenced instance.
* Script editor: class and method navigation combobox can now handle multiline method signatures
* Script editor: Bugfix in cut() method if cutting without active selection or if the last line of the selection was an empty line
* Fixes and improvements due to **deepcode.ai** analysis
* Bugfix in pythonWorkspace when parsing a class that has a __slots__ attribute. __slots__ can return either a list or tuple.
* Bugfix in uiOrganizer (QMetaType 43 exception due to exception of invoked method) in several methods if ui window or dialog have already been deleted before.
* Bugfix in node base structure of plots: input parameter 'dataObject' was wrongly added as output parameter to AbstractDObjPCLFigure
* Bugfixes and code refinement of UserOrganizer, UserModel If the startup argument name=<userID> is passed, this user is loaded without further dialog (only if password is required). New startup argument run=<path_to_script> is added (can be stacked) to indicate scripts that should be executed at startup
* Fixes and improvements of code editor, especially if tabs are used instead of spaces
* Many linux bugfixes, especially for a better compilation on a Raspberry
* Many other bugfixes

**Others:**

* Long error messages (sent to std::cerr) are now split into several lines based on word-boundaries
* Python syntax highlighting: added 'async' and 'await' as further keywords
* Documentation improved in many pages, added the section 'Contributing' with infos about a CMake style guide and information about translations.
* more unittests added
* many smaller improvements when using auto completion, calltips etc. from the python package jedi. Adaptions to support jedi 0.15, 0.16 and 0.17.
* templates for plugins adapted to the current state of the art
* replace unknown NULL macro by nullptr if the compiler is configured to C++11 e.g. via CMake: set(CMAKE_CXX_STANDARD 11) set(CMAKE_CXX_STANDARD_REQUIRED ON)

**Internal changes and improvements:**

* Official name of python class itom.pythonStream renamed to 'itom.pythonStream' (was pythonStream.PythonStream before)
* initAddIn in addInManager is now able to properly catch exceptions from constructors of plugins. initAddIn of actuators and dataIO merged into one common templated method.
* plotLegends is not used any more (since a long time) because it is integrated in itomWidgets. Therefore plotLegends is removed and also deleted from the SDK folder.
* Major refactoring of itomCommonPlotLib library: classes Channel, AbstractNode, AbstractFigure refactored and commented, parameter propagation through node tree improved, private classes added to enable easier updates with kept binary compatibility. These changes require adaptions in figure classes / plugins.
* Improved error message, containing all allowed enumeration values and keys, if one tries to set an enumeration or flag-based property (e.g. of a plot or another widget)


Plugins
******************

**Version 4.0.0 (2020-05-28)**

(136 commits in plugins repository)

* all plugins: 
    - adaptations for OpenCV 4
    - ``CMakeLists.txt`` and ``\*.cmake`` files adapted to (new) cmake style guide of itom 
      (see documentation/13_contributing/cmake_style_guide.rst). General settings 
      of itom libraries as well as plugin and   designer plugin libraries are now 
      added in the INIT_ITOM_LIBRARY() macro of ItomBuildMacros.cmake. Include 
      ItomBuildMacros.cmake at the beginning of the file and call INIT_ITOM_LIBRARY, 
      such that also CMake policies are globally set. ITOM_SDK_DIR is now 
      "auto"-detected in the overall CMakeLists.txt file.
    - adapted to new CMake macros / structure of itom SDK
    - Qt4 support removed


* new plugins: NerianSceneScanPro


* DummyMotor: status is updated after movement has been interrupted. Interrupt flag is always reset before a movement is started.
* GenICam: build script prepared for GenICam 3.1
* MSMediaFoundation: CPU load decreased during acquisition (> 50%) + minor bugfixes
* MSMediaFoundation: plugin refactored: static instances are mostly replaced by shared pointers to avoid for instance a crash when closing itom (fixes issue #7)
* NIDAQmx: implement stop() method of ito::AddInDataIO to stop a running task
* NIDAQmx: single value acquisition and write (software trigger only) is now possible. Added a demo for this.
* NerianSceneScanPro: removed parameter integrationTime... instead use manualExposureTime
* NiDAQmx: analog and digital input tasks now work in finite and continuous mode
* OpenCVFilters: added bilateralFilter for input image of type uint8, float32 because this filter is only implemented for those two types in OpenCV (4.1.2)
* OpenCVGrabber: avoid OpenCV warnung: "[ WARN:0] terminating async callback"
* PGRFlyCapture: added strobe_mode
* Roughness: Corrected Zsk and Zku parameters and changed Zv to be positive.
* SerialIO: add serialIO Sync Mutex
* Ximea: inserted aperture value as parameter
* Ximea: ready for hyperspectral sensors
* another time a deepcode.ai analysis
* demoAlgorithms: Extension of the demoAlgorithms with respect to the algorithmInterrupt branch of the itom sources: Added a long-running algorithm (10sec) 'demoCancellationFunction', that can be interrupted by a python scipt cancellation. Additionally, you can obtain progress and cancel this function via a GUI. See the widget example 'demoCancellationFunctionWidget' for this
* more work on niDAQmx: digital output tasks are now also running in finite and continuous mode
* niDAQmx: complete set of demo files for digital and analog input and output tasks (finite, continuous and single value)
* niDAQmx: rework of NIDAQMX plugin, mainly done by D. Nessett. See: https://bitbucket.org/dnessett/plugins
* niDAQmx: start trigger added
* niDAQmx: startDevice / stopDevice are now required for data acquisition or data output (similar to cameras). First steps towards continuous task. Finite analog / digital input are working.
* niDAQmx: working analog and digital, input and output, finite and continuous tasks; toolbox dock widget added; tdms logging for input tasks possible
* work on analog output tasks for niDAQmx


Designer Plugins
******************

**Version 4.0.0 (2020-05-28)**

(more than 58 commits in designerPlugins repository)

* general: ``CMakeLists.txt`` and ``\*.cmake`` files adapted to (new) cmake style 
  guide of itom (see documentation/13_contributing/cmake_style_guide.rst). 
  General settings of itom libraries as well as plugin and designer plugin libraries 
  are now added in the INIT_ITOM_LIBRARY() macro of ItomBuildMacros.cmake. Include 
  ItomBuildMacros.cmake at the beginning of the file and call INIT_ITOM_LIBRARY, 
  such that also CMake policies are globally set. ITOM_SDK_DIR is now 
  "auto"-detected in the overall CMakeLists.txt file.
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

