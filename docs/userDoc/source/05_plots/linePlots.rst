line plots (1D)
******************


"Itom1DQwtPlot" is the basic line plot for plotting 'dataObjects' with shape [1xM] based on the Qwt library.
All DataTypes are accepted. To plot complex objects, it is possible to select between the following modes: "absolut", "phase", "real" and "imaginary".
RGBA-Values are currently interpreted as Gray-Values using RGBA as gray.

The plot allows value and min/max-picking via place-able marker.

The plot supports geometric element and marker interaction via **drawAndPickElements(...)** and **call("userInteractionStart",...)**. See section :ref:`primitives` for a short introduction.

The plot does not support z-stack sectioning at the momtent. 

You can also use the "matplotlib"-backend to plot slices or xy-coordinates. See section :ref:`pymod-matplotlib` for more information about how to use "matplotlib".

The plot-canvas can be exported to vector and bitmap-graphics via button or menu entry or it can be exported to clipBoard via ctrl-c or a plublic slot.

Properties
---------------

**selectedGeometry** : *int*, get the currently selected geometric element within this plot

**enablePlotting** : *bool*, enable and disable internal plotting functions and GUI-elements for geometric elements.

**keepAspectRatio** : *bool*, enable and disable a fixed 1:1 aspect ratio between x and y axis.

**geometricElementsCount** : *int*, number of currently existing geometric elements.

**geometricElements** : *ito::DataObject*, geometric elements defined by a float32[11] array for each element.

**axisFont** : *QFont*, font for axes tick values.

**labelFont** : *QFont*, Font for axes descriptions.

**titleFont** : *QFont*, Font for title.

**valueLabel** : *QString*, Label of the value axis (y-axis) or '<auto>' if the description should be used from data object.

**axisLabel**: *QString*, Label of the direction (x/y) axis or '<auto>' if the descriptions from the data object should be used.

**title** : *QString*, Title of the plot or '<auto>' if the title of the data object should be used.

**bounds** : *QVector<QPointF>*, 

**colorMap** : *QString*, Color map (string) that should be used to colorize a non-color data object.

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
 
 Set new source object to this plot. Usually invoked by any camera if used as a live image.  

 *source, ito::DataObject *: The new dataObject to display
 
 *semaphore, ItomSharedSemaphore*: A semaphore to handle the multi-threading.
 
 
**refreshPlot( )**: 

 Refresh / redraw current plot

**copyToClipBoard()**:
 
 Copy current canvas with white background to clipBoard


Deprecated figures
==========================
 
The plot-dll "itom1DQWTFigure"  is deprecated and has been replaced by  "Itom1DQwtPlot".