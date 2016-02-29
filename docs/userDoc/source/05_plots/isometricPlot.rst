isometric Plot
****************

"ItomIsoGLWidget" is a plot for pseudo 3D visualization of image like DataObjects. It is based on openGL and renders the objects
either to triangles ("triangle mode") or points ("Joe-Mode").
All DataTypes except "rgba32" are accepted. To plot complex objects, it is possible to select between the following modes: "absolute", "phase", "real" and "imaginary".

The figure does not support z-stack sectioning. The "ItomIsoGLWidget" does support neither graphic element / marker plotting nor line or pixel picking. Hence this plot will be improved and replaced by a new version for the next release.

Properties
=================
 
**colorMap** : *QString*, Defines which color map should be used [e.g. grayMarked, hotIron].

**zAxisInterval** : *QPointF*, Sets the visible range of the displayed z-axis (in coordinates of the data object) or (0.0, 0.0) if range should be automatically set [default]. **Not implemented yet**

**yAxisInterval** : *QPointF*, Sets the visible range of the displayed y-axis (in coordinates of the data object) or (0.0, 0.0) if range should be automatically set [default]. **Not implemented yet**

**xAxisInterval** : *QPointF*, Sets the visible range of the displayed x-axis (in coordinates of the data object) or (0.0, 0.0) if range should be automatically set [default]. **Not implemented yet**

**camera** : *ito::AddInDataIO*, Use this property to set a camera/grabber to this plot (live image).

**displayed** : *ito::DataObject*, This returns the currently displayed data object [read only].

**source** : *ito::DataObject*, Sets the input data object for this plot.

**contextMenuEnabled** : *bool*, Defines whether the context menu of the plot should be enabled or not. **Not implemented yet**

**toolbarVisible** : *bool*, Toggles the visibility of the toolbar of the plot.  **Not implemented yet**
 
Slots
=================

**ito::RetVal setLinePlot( double x0, double y0, double x1, double y1 [, int linePlotIdx = -1])**:

 this can be invoked by python to trigger a line plot, inherited from *class AbstractDObjFigure*, **not implemented at the moment**

 *x0, double*: first position of line plot in x-Direction
 
 *y0, double*: first position of line plot in y-Direction
 
 *x1, double*: second position of line plot in x-Direction
 
 *y1, double*: second position of line plot in x-Direction
 

**ito::RetVal setSource( ito::DataObject source, ItomSharedSemaphore* )**
 
 Set new source object to this plot. Usually invoked by any camera if used as a live image from **internal C++-Code**.  

 *source, ito::DataObject *: The new dataObject to display
 
 *semaphore, ItomSharedSemaphore*: A semaphore to handle the multi-threading.

 
**refreshPlot( )**: 

 Refresh / redraw current plot 

**triggerReplot( )**: 

 Refresh / redraw current plot  
 
Deprecated figures
==========================

None