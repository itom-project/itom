.. include:: ../include/global.inc

.. _primitives:

Markers, user selections and geometric shapes
**************************************************

The plot-widgets :ref:`itom1dqwtplot` and :ref:`itom2dqwtplot` (short: Qwt-plots) support displaying markers, geometric shapes by script or by user interaction.
This section will give a short introduction about plotting, read- /write-functions and the corresponding plots and the internal geometric element structure.

At last the evaluateGeomtrics-plugin for direct evaluation of geometric elements is introduced.

Displaying geometric shapes or markers onto Qwt-plots can be done by three different ways:

* GUI based approach by draw and delete options in the toolbar or the **Tools** menu.
* Displaying shapes and markers by a Python script.
* Get the position and size of specified shapes or markers by the user and optionally display them by script.

Drawing items into a Qwt-plot
==================================

The plot functionality can be accessed by three different ways. The first way is the GUI based approach where the user presses the
"switch draw mode"-button in the button-bar of the plot. The button represents the current item to be plotted.
The red X ("clear button") will delete all geometric elements within the plot either drawn by hand or by script.

.. figure:: images/drawInto2DToolbarIcons.png
    :width: 893
    :align: center

At the moment "itom" supports the following shapes:

* point
* line
* rectangle (alignment parallel to axes)
* ellipse (alignment parallel to axes)

To draw an item, select the right shape from the **tools >> switch draw mode** menu. The button is then in an active, toggled state and you can start
by clicking the canvas. For lines, rectangles and ellipses, you have to click once for the first point of the item and another time for the second time.
During the design mode, a green line indicates the current position and size of the draft, after the last necessary mouse click, the shape design
is finished and its color turns to the corresponding inverse color of the current color palette.

.. figure:: images/drawInto2D.png
    :width: 893
    :align: center

By clicking the **tools >> clear marker** menu, all current shapes are deleted. After creation the geometric elements can be edited by left-clicking one of the element handles which becomes high-lighted (squares) and moving the mouse.
By pressing the "ctrl"-button during mouse-movement the element resize behaviour will be changed depending on the element type.
Lines will be changed to horizontal or vertical alignment.
Rectangles and ellipses will be become squares or circles according to plot coordinates (x/y-space) and not pixel coordinates.
To avoid confusion with plot aspect, a button for fixed axis aspect ratio ("1:1") was added to the plot bar.

Script based pick and plot from / to a Qwt-widget
====================================================================

To allow more complex user interaction with scripts, e.g. script based element picking, the plot functionality can be started by script either blocking or non-blocking.

.. code-block:: python

    myImage = dataObject.randN([200, 200], 'float32')
    [number, handle] = plot(myImage, "itom2dQwtPlot")

    # Blocking access which return the values for a single point in myElement
    shape = handle.drawAndPickElements(plotItem.PrimitivePoint, 1)
    # get center of shape
    shape[0].center

    # None blocking plot
    # Structure will be dataObject([1, 11], 'float32') with [idx, type, x, y, 0, 0, 0, 0]
    handle.call("userInteractionStart", plotItem.PrimitivePoint, True, 1)

    # --> Read out later after plot is finished
    myGeometry = handle["geometricShapes"]


The blocking code will wait until the selection is done or the selection was aborted by user and will than return the corresponding object.
The non-blocking code will return directly. To access the geometric elements the corresponding "signal" for userInteractionDone should be used to noticed
the end of the user interaction.

The geometric elements can also be set by script by calling the corresponding slot.

.. code-block:: python

    obj = dataObject.zeros([1000,1000])
    [nr,h] = plot(obj, "itom2dqwtplot")
    element=shape(shape.Point,[50,50])
    h['geometricShapes']=element

For more information see also :ref:`plot-markers`

The geometric elements can be read any time using the property "geometricShapes".

.. code-block:: python

    # Reading geometric elements
    myGeometry = handle["geometricShapes"]

The object "myGeometry" consists of all geometric elements with in the plot.
This kind of reading differs from the blocking-variant ("drawAndPickElements"). The blocking-variant returns only the data created during the current function call and ignores old geometric elements.
For the different definitions of the geometric elements see section "Indexing of Geometric Elements".

Implemented Functions, Signals and Slots
===================================================

The Qwt-plot widgets functions had to be updated. The Qwt-Widgets got the following properties, respectively setter- / getter-functions related to plotting:

*  "geometricElementsCount",    get the number of geometric elements in this plot, READONLY
*  "keepAspectRatio",           enable or disable a fixed aspect ratio of 1:1 for the plot canvas.
*  "enablePlotting",            enable and disable plotting toolbar
*  "selectedGeometry",          set / get the current selected geometric element

In complete these the functionality of the drawing interface, the following slots have been added to the widgets:

*  "plotMarkers"                Add markers and geometric elements to plot according to dataObject and style type. For geometric "style is not used", type in "b".
*  "userInteractionStart"       Start a non-blocking user interaction for "multipointpick" or geometric elements with element count.
*  "clearGeometricElements"     Delete all existing geometric elements within the plot

To register changes in the plot elements and finished user interactions, the following signals where implemented:

*  "userInteractionDone",       Emitted, when user interaction is finished or aborted
*  "plotItemChanged",           Emitted, when a geometric element was changed and changing is finished. Not fully Python-Compatible.
*  "plotItemDeleted",           Emitted, when a specific plot item is deleted
*  "plotItemsDeleted",          Emitted, when all plot items are deleted
*  "plotItemsFinished",         Emitted, when the plotting function is finished, similar to userInteractionDone

For the blocking connection the plotItem-class got the additional function drawAndPickElements(type, dataObject, count), see :py:class:`~itom.plotItem`.

.. note::

    If the plot is embedded in a graphical user interface, the python based access to the plot via its object name returns an instance of :py:class:`~itom.uiItem`.
    However, you can cast this instance to :py:class:`~itom.plotItem` using a python cast operator:

    .. code-block:: python

        import itom
        plot = itom.plotItem(myGui.plotObjectName)

    If *myGui.plotObjectName* is not an instance of :py:class:`~itom.plotItem` a runtime error is thrown. This cast is only available for itom > 1.4.0.

Indexing for Geometric Elements
==================================

The definition of the geometric elements depends on the implementation. The "plotMarker" and its corresponding getter- / setter-function uses a Matlab orientated structure.
The structure a dataObject with 8 rows and n columns where n depends on the number of elements.
Points are defined by their location, while ellipses and rectangles are defined by their diagonal edges.

The "geometricElements"-property uses geometric elements in a more mathematical orientated description. The dataObject structure is defined as "float32" with n by 11 elements.
Most setter functions also support "float32" elements. Each of the n rows corresponds to an elements (except polygon-shapes). The indexing follows the geometricPrimitive-struct in c++.

The geometricPrimitive is a struct within the c-Stuctur of the program designed for exchanging the geometric elements from plots to other elements.
The structure can be used row-wise as dataObject or float32-lists

At the moment only tPoint, tLine, tEllipse and tRectangle are supported.

The cells contain:

1. The unique index of the current element, castable to int32 with a maximum up to 16bit index values

2. Type flag 0000FFFF and further flags e.g. read&write only FFFF0000

3. First coordinate with x value

4. First coordinate with y value

5. First coordinate with z value


All other values depends on the element type and may change between each type.

* A point is defined as idx, flags, centerX0, centerY0, centerZ0
* A line is defined as idx, flags, x0, y0, z0, x1, y1, z1
* A ellipse is defined as idx, flags, centerX, centerY, centerZ, r1, r2
* A circle is defined as idx, flags, centerX, centerY, centerZ, r
* A rectangle is defined as idx, flags, x0, y0, z0, x1, y1, z1, alpha
* A square is defined as idx, flags, centerX, centerY, centerZ, a, alpha
* A polygon is defined as idx, flags, posX, posY, posZ, directionX, directionY, directionZ, idx, numIdx

.. toctree::
   :hidden:

.. doxygenclass:: ito::Shape::ShapeType
    :project: itom
    :members:


Evaluation of Geometric Elements
==================================

The evaluateGeomtrics-widget is designed to load geometric definition stored in a float32 dataObject with a column-size of >10 elements and a row for each geometric element to display.
Further more it allows the evaluation of geometric relations between the geometric primitives. See section :ref:`listCustomDesignerWidgets` for the widget description.


Demo Scripts and Examples
==================================

- **uiMeasureToolMain.py**

  | *Description*: Advanced GUI which enables geometric plotting and measurements within a 2D-QWT-Plot. This file shows how to auto-connect to signals and how to use buttons. The corresponding ui-file is uiMeasureToolMain.ui.

- **demoPickPointsAndMarkers.py**

  | *Description*: Demo for picking & plotting points and picking & plotting ellipses.
