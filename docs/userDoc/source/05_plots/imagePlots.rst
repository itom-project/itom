2D image plots
****************

"Itom2dQwtPlot" and "GraphicViewPlot" are the basic plots for visualization of images, dataObjects or other array-like objects.
Both plots have a line-cut and point picker included. By pressing "Ctrl" during picker movement, the picker can only be moved 
horizontal or vertical according to the mouse movement.

You can also use the "matplotlib"-backend to plot any data structures (lines, bars, statistical plots, images, contours, 3d plots...). 
See section :ref:`pymod-matplotlib` for more information about how to use "matplotlib".

Itom2dQwtPlot
==========================

"Itom2dQwtPlot" is designed for visualizing metrical data, false color or topography measurements.
It supports the axis-scaling / axis offset of **dataObjects**, offers axis-tags and meta-data handling.
All data types are accepted except the plotting of real color objects (rgba). To plot complex objects, it is possible to choose 
between the following modes: "absolut", "phase", "real" and "imaginary". The data is plotted mathematically correct. This means 
the value at [0,0] is in the lower left position. This can be changed by the property *yAxisFlipped*.

The plot supports geometric element and marker interaction via **drawAndPickElements(...)** and **call("userInteractionStart",...)**. 
See section :ref:`primitives` for a short introduction.

**Features:**

* Export graphics to images, pdf and vector graphics.
* Metadata support (the 'title'-tag is used as title of the plot).
* Supports fixed ratio x/y-axis but not necessary fixed ratio to monitor-pixel
* Drawing of geometrical elements and markers by script and user interaction.
* Images are displayed either mathematically ([0,0] lower left) or in windows-style ([0,0] upper left) (Property: 'yAxisFlipped')

Properties
---------------
**selectedGeometry** : *int*, Get or set the currently highlighted geometric element. After manipulation the last element stays selected.

**showCenterMarker** : *bool*, Enable a marker for the center of a data object.

**enablePlotting** : *bool*, Enable and disable internal plotting functions and GUI-elements for geometric elements.

**keepAspectRatio** : *bool*, Enable and disable a fixed 1:1 aspect ratio between x and y axis.

**geometricElementsCount** : *int, Number of currently existing geometric elements.

**geometricElements** : *ito::DataObject*, Geometric elements defined by a float32[11] array for each element.

**axisFont** : *QFont*, Font for axes tick values.

**labelFont** : *QFont*, Font for axes descriptions.

**titleFont** : *QFont*, Font for title.

**colorMap** : *QString*, Defines which color map should be used [e.g. grayMarked, hotIron].

**colorBarVisible** : *bool*, Defines whether the color bar should be visible.

**valueLabel** : *QString*, Label of the value axis or '<auto>' if the description should be used from data object.

**yAxisFlipped** : *bool*, Sets whether y-axis should be flipped (default: false, zero is at the bottom).

**yAxisVisible** : *bool*, Sets visibility of the y-axis.

**yAxisLabel** : *QString*, Label of the y-axis or '<auto>' if the description from the data object should be used.

**xAxisVisible** : *bool*, Sets visibility of the x-axis.

**xAxisLabel** : *QString*, Label of the x-axis or '<auto>' if the description from the data object should be used.

**title** : *QString*, Title of the plot or '<auto>' if the title of the data object should be used.

**zAxisInterval** : *QPointF*, Sets the visible range of the displayed z-axis (in coordinates of the data object) or (0.0, 0.0) if range should be automatically set [default].

**yAxisInterval** : *QPointF*, Sets the visible range of the displayed y-axis (in coordinates of the data object) or (0.0, 0.0) if range should be automatically set [default].

**xAxisInterval** : *QPointF*, Sets the visible range of the displayed x-axis (in coordinates of the data object) or (0.0, 0.0) if range should be automatically set [default].

**camera** : *ito::AddInDataIO*, Use this property to set a camera/grabber to this plot (live image).

**displayed** : *ito::DataObject*, This returns the currently displayed data object [read only].

**source** : *ito::DataObject*, Sets the input data object for this plot.

**contextMenuEnabled** : *bool*, Defines whether the context menu of the plot should be enabled or not.

**toolbarVisible** : *bool*, Toggles the visibility of the toolbar of the plot.


Signals
---------------

**plotItemsFinished(int,bool)**: Signal emitted if plotting of n-elements if finished. Use this for non-blocking synchronisation.

 *counts, int*: Number of plotted elements

 *aborted, bool*: Flag showing if draw function was cancled during plotting
 
 
**plotItemsDeleted()**: 
 
 Signal emitted if geometric elements were deleted.

 
**plotItemDeleted(ito::int32)**: 
 
 Signal emitted if specified geometric element was deleted.

 
**plotItemChanged(ito::int32,ito::int32,QVector<ito::float32>)**: 
 
 Signal emitted if specified geometric element was changed.

 *idx, ito::int32*: Index of changed element

 *element, QVector<ito::float32>*: New geometric featured of changed element

 
**userInteractionDone(int,bool,QPolygonF)**: 
 
 Signal emitted if user interaction is done. Internal function used for blocking synchronisation.
 

Slots
---------------

**ito::DataObject getDisplayed( )**:

 Retrieve currently displayed dataObject. 

 
**ito::RetVal clearGeometricElements( )**:

 Delete all geometric Elements


**void userInteractionStart( int type, bool start [, int maxNrOfPoints = -1] )**: 

 This slot should be called of non-blocking GUI-based drawing of geometric elements within this plot is necessary. See section :ref:`primitives` for a short introduction.

 *type, int*: type to plot
 
 *start, bool*: true if plotting should be started
 
 *maxNrOfPoints, int*: number of elements to plot

 
**ito::RetVal deleteMarkers( int id)**: 
 
 Delete geometric element

 *id, int*: the 0-based index of specific geometric element
 
 
**ito::RetVal deleteMarkers( QString id)**:

 Delete point based marker

 *id, QString*: the name based identifier of specific geometric element

 
**ito::RetVal plotMarkers( ito::DataObject coords, QString style [, QString id = "" [, int plane = -1]])** :
 
 This slot is called to visualize markers and python-based plotting of geometric elements within this plot. See section :ref:`primitives` for a short introduction.
 
 *coords, ito::DataObject*: an initilized dataObject with a column per element and a set of rows describing its geometric features
 
 *style, QString*: Style for plotted markers, for geometric elements it is ignored
 
 *id, QString*: Text based id for markers will be ignored for geometric elements.
 

**ito::RetVal setLinePlot( double x0, double y0, double x1, double y1 [, int linePlotIdx = -1])**:

 this can be invoked by python to trigger a lineplot, inherited from *class AbstractDObjFigure*

 *x0, double*: first position of linePlot in x-Direction
 
 *y0, double*: first position of linePlot in y-Direction
 
 *x1, double*: second position of linePlot in x-Direction
 
 *y1, double*: second position of linePlot in x-Direction
 

**ito::RetVal setSource( ito::DataObject source, ItomSharedSemaphore* )**
 
 Set new source object to this plot. Usually invoked by any camera if used as a live image from internal C++-Code. 

 *source, ito::DataObject *: The new dataObject to display
 
 *semaphore, ItomSharedSemaphore*: A semaphore to handle the multi-threading.
 
 
**refreshPlot( )**: 

 Refresh / redraw current plot
 

GraphicViewPlot
==========================

"GraphicViewPlot" is designed for the fast display of images, e.g. direct grabber output or colored images. 
It allows plotting real colors (at the moment only 24-bit or 32-bit stored as int32 or RGBA32). It does not handle meta-data.
All DataTypes are accepted. To plot complex objects, it is possible to select between the following modes: "absolut", "phase", "real" and "imaginary".
The data is plotted image orientated. This means the value at [0,0] is in the upper left position.

The figure allows z-stack sectioning. An automatic video-like visualisation is in preparation.

The "GraphicViewPlot" does not support graphic element / marker plotting. Use "Itom2dQwtPlot" instead for this case.

Features:

* Supports real color and gray-value visualization
* Supports fixed ratio between image-pixel and monitor-pixel (4:1 - 1:4)
* Fast implementation for 8-bit and 16-bit direct camera output.
* Images are displayed in windows-style 

Properties
---------------

**colorMap** : *QString*, Color map (string) that should be used to colorize a non-color data object [e.g. grayMarked, hotIron].

**colorBarVisible** : *bool*, Defines whether the color bar should be visible.

**colorMode** : *int*, Defines color handling, either "palette-based color" or "RGB-color"

**zAxisInterval** : *QPointF*, Sets the visible range of the displayed z-axis (in coordinates of the data object) or (0.0, 0.0) if range should be automatically set [default].

**yAxisInterval** : *QPointF*, Sets the visible range of the displayed y-axis (in coordinates of the data object) or (0.0, 0.0) if range should be automatically set [default].

**xAxisInterval** : *QPointF*, Sets the visible range of the displayed x-axis (in coordinates of the data object) or (0.0, 0.0) if range should be automatically set [default].

**camera** : *ito::AddInDataIO*, Use this property to set a camera/grabber to this plot (live image).

**displayed** : *ito::DataObject*, This returns the currently displayed data object [read only].

**source** : *ito::DataObject*, Sets the input data object for this plot.

**contextMenuEnabled** : *bool*, Defines whether the context menu of the plot should be enabled or not.

**toolbarVisible** : *bool*, Toggles the visibility of the toolbar of the plot.
 

Slots
---------------

**ito::RetVal setLinePlot( double x0, double y0, double x1, double y1 [, int linePlotIdx = -1])**:

 this can be invoked by python to trigger a lineplot, inherited from *class AbstractDObjFigure*, not implemented at the moment

 *x0, double*: first position of linePlot in x-Direction
 
 *y0, double*: first position of linePlot in y-Direction
 
 *x1, double*: second position of linePlot in x-Direction
 
 *y1, double*: second position of linePlot in x-Direction
 

**ito::RetVal setSource( ito::DataObject source, ItomSharedSemaphore* )**
 
 Set new source object to this plot. Usually invoked by any camera if used as a live image from internal C++-Code.  

 *source, ito::DataObject *: The new dataObject to display
 
 *semaphore, ItomSharedSemaphore*: A semaphore to handle the multi-threading.

 
**refreshPlot( )**: 

 Refresh / redraw current plot 

Signals
---------------

No public signals at the moment.
 
Deprecated figures
==========================

The plot-dll "itom2DQWTFigure" and "itom2DGVFigure" are deprecated and have been replaced by  "Itom2dQwtPlot" and "GraphicViewPlot".