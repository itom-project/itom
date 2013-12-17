.. include:: ../../include/global.inc

.. _primitives:

**Primitives** - Marking and Measuring
==============================================

The plot-widgets itom1DQwtPlot and itom2DQwtPlot supports plotting of geometric primitives by user interaction and script language.
This section will give a short introduction about ploting, read- /write-functions and the correspondig plots and the internal geometric element structure.

At last the evaluateGeomtrics-plugin for direct evaluation of geometric elements is introduced.

Drawing items into a QWT-Plot
----------------------------------------------

The plot functionality can be accessed by three different ways. The first way is the GUI based approach by which the user presses the 
"switch draw mode"-button in the button-bar of the plot. The button represents the current item to be plotted. 
The red X ("clear button") will delete all geometric elements within the plot either drawn by hand or by script.

.. figure:: images/drawInto2DToolbarIcons.png
    :scale: 50%

At the moment "itom" only supports "point", "line", "rectanlge" and "ellipse" but further items, e.g. "circle" and "polygons", are in preparation. 
To draw an item simply click into the image space and left-click the mouse. 
In case of elements with at least more than a marker, you can now set the size of the element by setting the second point by left-clicking again.
During plotting a green lined geometric element appears. After finishing the element color turns to the first inverse 
color of the current color palette with handles (diamonds or sqares) colored with the second inverse color of the current palette.
    
.. figure:: images/drawInto2D.png
    :scale: 50%

After creation the geometric elements can be editied by left-clicking one of the element handles  which becomes high-lighted (squares) and moving the mouse.
By pressing the "ctrl"-button during mouse-movement the element resize behavior will be changed depending on the element type. 
Lines will be changed to horizontal or vertical alignment.
Rectangles and ellipses will be become squares or circles according to plot coordinates (x/y-space) and not pixel coordinates. 
To avoid confusion with plot aspect, a button for fixed axis aspect ratio ("1:1") was added to the plot bar.

To allow more complex user interaction with scripts, e.g. script based element picking, the plot functionality can be started by script either blocking or non-blocking.

.. code-block:: python
    
    myImage = dataObject.randN([200, 200], 'float32')    
    [number, handle] = plot(myImage, "itom1dQwtPlot")
    
    # Blocking access which return the values
    myElement = dataObject()
    myBlocking Connection
    
    # None blocking plot
    
The blocking code will wait until the selection is done or the selection was aborted by user and will than return the corresponding object.
The non-blocking code will return directly. To access the geometric elements the corresponding "signal" for userInteractionDone should be used to noticed
the end of the user interaction.

The geometric elements can also be set by script by calling the corresponding slot.

.. code-block:: python
    
    # the magic ploting commane
    
    myPlot.call("blabla")

--> ToDo: Input Object describtion 

Reading from a QWT-Plot
----------------------------------------------

The geometric element can be read any time calling the slot ""

.. code-block:: python
    
    # the magic ploting commane
    
    myPlot.call("blabla")

--> ToDo: Output-Object describtion 

The object consists of all geometric elements with in the plot. Each row corresponds to one geometric element while the parameters for each element are align column-wise.

This kind of reading differs from the blocking-variant. The blocking-variant returns only the data created during the current function call and ignores old geometric elements.
In this case the elements are aligned column-wise. This means each column corresponds to on element while its data is stored along the rows.

Signals and Slots
----------------------------------------------



Indexing for Geometric Elements
----------------------------------------------

The geometricPrimitives is a struct within the c-Stuctur of the programm used for exchanging the geometric elements from plots to other elements.
The structur can be used rowise as dataObject or float32-lists

At the moment only tPoint, tLine, tEllipse and tRectangle are supported.

The cells contain:

1. The unique index of the current primitive, castable to int32 with a maximum up to 16bit index values

2. Type flag 0000FFFF and further flags e.g. read&write only FFFF0000

3. First coordinate with x value

4. First coordinate with y value

5. First coordinate with z value
    

All other values depends on the primitiv type and may change between each type.

* A point is defined as idx, flags, centerX0, centerY0, centerZ0
* A line is defined as idx, flags, x0, y0, z0, x1, y1, z1
* A ellipse is defined as idx, flags, centerX, centerY, centerZ, r1, r2
* A circle is defined as idx, flags, centerX, centerY, centerZ, r
* A rectangle is defined as idx, flags, x0, y0, z0, x1, y1, z1, alpha
* A square is defined as idx, flags, centerX, centerY, centerZ, a, alpha
* A polygon is defined as idx, flags, posX, posY, posZ, directionX, directionY, directionZ, idx, numIdx

.. toctree::
   :hidden:

.. doxygenclass:: ito::PrimitiveContainer
	:project: itom
	:members:
    
    
Evaluation of Geometric Elements 
----------------------------------------------

The evaluateGeomtrics-widget is designed to load geometric definition stored in a float32 dataObject with a column-size of >10 elments and a row for each geometric element to display.
Further more it allows the evaluation of geometric relations between the geometric primitives.
