.. include:: ../include/global.inc

.. |mm| unicode:: U+00B5 m

Changelog Version 3.2.0
#########################

itom
********

**Version 3.2.0 (2019-04-07)**

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
* addded :py:meth:`itom.dataIO.getExecFuncsList` and :py:meth:`itom.actuator.getExecFuncsList` to print a list of available **exec-functions** of a specific plugin instance to the console.
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
* performance improvements in console widget, if a lot of text should be printed: added an internal intermediate buffer, wrap long lines (optional) into several lines, each starting with ``... ``.
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
******************

**Version 3.2.0 (2019-04-07)**

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
******************

**Version 3.2.0 (2019-04-07)**

(more than 89 commits in designerPlugins repository)

New features: support matplotlib 3.x including new *edit parameters* dialog, volumeCut in itom2dqwtplot, color of shapes can be assigned indivually (in script) 

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
