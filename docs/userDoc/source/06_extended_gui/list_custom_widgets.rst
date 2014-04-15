.. include:: ../include/global.inc

.. _listCustomDesignerWidgets:

Custom Designer Widgets
==============================================

Beside QtDesginer-Widgets for plots and figures (see  :ref:`PlotsAndFigures`) some non-plotting widgets have been develop to give the user GUI-based access to itom specific objects and functions.
The openSource-Widgets are:

* DataObjectTable
* dObMetaDataTable
* EvaluateGeometricsFigure
* MotorController

These widgets can be used like any other type of widget within an ui-dialog with 2 exceptions:

1. The ui-Dialog must be loaded and initilized within a itom-python context (e.g. script in itom).

2. Some properties are not accesable (DESIGNABLE) in the QtDesigner (e.g. actuator-handles) and must be set during initilization in itom-python.

To add such a widget to your ui-file, you can drag&drop them in the QtDesigner like any other widget.  


DataObjectTable
----------------------------------------------

The "DataObjectTable" can be used to visualize or edit a dataObject in a table based widget like in "Matlab". The widget is not inherited from a AbstractDObject and can not be used for a live plot.

Properties
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**data**: *ito::DataObject*, the dataObject to be shown

**readOnly**: *bool, DESIGNABLE*, enable write protection

**min**: *double, DESIGNABLE*, get/set minimum value

**max**: *double, DESIGNABLE*, get/set maximum value

**decimals**: *int, DESIGNABLE*, number of decimals to be shown within each cell

**defaultCols**: *int, DESIGNABLE*, number of column to be shown

**defaultRows**: *int, DESIGNABLE*, number of rows to be shown

**horizontalLabels**: *QStringList, DESIGNABLE*, list with labels for each column row

**verticalLabels**: *QStringList, DESIGNABLE*, list with labels for each shown row



dObMetaDataTable
----------------------------------------------

The "dObMetaDataTable" can be used to visualize the metaData of a dataObject, e.g. to realize a measurement protocol . The widget is not inherited from a AbstractDObject and can not be used for a live plot.

Properties
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**data**: *ito::DataObject*, the dataObject to be shown

**readOnlyEnabled**: *bool, DESIGNABLE*, enable write protection

**detailedInfo**: *bool, DESIGNABLE*, Toogle between basic and detailed metaData

**previewEnabled**: *bool, DESIGNABLE*, Add a small quadratic image downsampled from the dataObject as a preview to the meta data.

**previewSize**: *int, DESIGNABLE*, Set the preview size in pixels,

**decimals**: *int, DESIGNABLE*, number of decimals to be shown within each cell

**colorBar**: *QString, DESIGNABLE*, the name of the color bar for the preview, *not implemented yet*


EvaluateGeometricsFigure
----------------------------------------------

The evaluateGeomtrics-widget is designed to load geometric definition stored in a float32 dataObject with a column-size of >10 elements and a row for each geometric element to display.
Further more it allows the evaluation of geometric relations between the geometric primitives. It contains a tableView and although is is inherited by AbstractDObject it should not be used for "liveVisualisation of dataObject".


Properties
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**title**: *QString, DESIGNABLE*, Title of the plot or '<auto>' if the title of the data object should be used. *Not implemented yet*
**valueUnit**: *QString, DESIGNABLE*, The value unit for the metrical calculations that is used within the plot.
**titleFont**: *QFont, DESIGNABLE*, Font for title. *Not implemented yet*
**labelFont**: *QFont, DESIGNABLE*, Font for labels. *Not implemented yet*
**relations**: *ito::DataObject*, Get or set N geometric elements via N x 11 dataObject of type float32. 
**relationNames**: *QStringList, DESIGNABLE*, A string list with the names of possible relation. The first elements [N.A., radius, angle to, distance to, intersection with, length and area] are read only and are calculated with these widget. For external calculated values you can define custom names e.g. roughness..
**destinationFolder**: *QString, DESIGNABLE*, Set a default export directory.
**lastAddedRelation**: *int, DESIGNABLE*, Get the index of the last added relation.
**considerOnly2D**: *bool, DESIGNABLE*, If true, only the x & y coordinates are considered.

Slots
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**ito::RetVal addRelation(ito::DataObject importedData)** 
 
 Add a relation to the current context. The relation must be 
 
 *importedData, ito::DataObject*: geometric element expressed by 1 x 4 dataObject of type float32. 
 
 
**ito::RetVal modifyRelation(const int idx, ito::DataObject relation)**

 Modify an existing relation addressed by the idx.

 *idx, int*: Index of relation to modify
 
 *relation, ito::DataObject*: geometric element expressed by 1 x 4 dataObject of type float32. 

 
**ito::RetVal addRelationName(const QString newName)**
 
 Add a new relation name to the relationNameList.

 *newName, QString*: new relation name to be appended
 
 
**ito::RetVal exportData(QString fileName, ito::uint8 exportFlag)**
 
 Export data to csv or xml

 *fileName, QString*: Destination file name
 
 *exportFlag, int*: Export flag, exportCSVTree  = 0x00, exportCSVTable = 0x01, exportXMLTree  = 0x02, exportCSVList  = 0x03, showExportWindow = 0x10
 
 
**ito::RetVal plotItemChanged(ito::int32 idx, ito::int32 flags, QVector<ito::float32> values)**

 Slot for direct connection between this widget and a plot (e.g. itom2dQwtPlot) to notify changes within the plotted geometry. Internal relations are automatically updated. External relation values (e.g. roughness) can not be updated automatically.
 
 *idx, int*: Index of the modified geometric element
 
 *flags, int*: Type (and meta properties) of geometric elements which was changed. If type differs from original type clear and refill is necessary.
 
 *values, QVector<ito::float32>*: Geometric parameters of the modified geometric item.
 
 
**ito::RetVal clearAll(void)**
 
 Clear all elements and relations in this plot. 


Signals
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

None


MotorController
----------------------------------------------

The "MotorController"-widget gives the user the some basic functions for generic motor positioning and position reporting within a complex GUI. The widget can be used for 1 to 6 axis and can be used readOnly or as a type of software control panel.
The widget updated in a fixed interval (can be deactivated). During measurements the widget should be disabled to avoid user errors. A support for the 3DConnexion-Mouse is planed but *Not implemented yet*.

The motor should support up *slot: RequestStatusAndPosition* and *signal: actuatorStatusChanged* for semaphore free communication, but this is not necessary.

Properties
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**actuator**: *ito::AddInActuator*, Handle to the actuator to be used, not DESIGNABLE

**numberOfAxis**: *int, DESIGNABLE*, Number of axis to be visible

**unit**: *QString, DESIGNABLE*, Base unit for spinboxes and movements, e.g. nm, micron, mm, m, km

**readOnly**: *bool, DESIGNABLE*, Toogle read only

**autoUpdate**: *bool, DESIGNABLE*, Toogle automatic motorposition update

**smallStep**: *double, DESIGNABLE*, Distances for the small step button, same value for plus and minus

**bigStep**: *double, DESIGNABLE*, Distances for the large step button, same value for plus and minus

**absRel**: *bool, DESIGNABLE*, Toogle between absolut or relative position display. Origin can be set via context menu.

**allowJoyStick**: *bool, DESIGNABLE*, Allow a software joystick, e.g. usb or gameport, not implemented yet.


Slots
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**void triggerActuatorStep(const int axisNo, const bool smallBig, const bool forward)**

 Trigger a step of axis *axisNo* with a distance either *bigStep (true)* or *smallStep (false)* and either *forward (true)* or *backwards (false)*
 
 
**void actuatorStatusChanged(QVector<int> status, QVector<double> actPosition)**

 Internal slot for c++-Code connected to the corresponding signal of the actuator. Do not call this from python.

 
**void triggerUpdatePosition(void)**
 
 Usually called by the internal timer, the context-menu or python to update the current motor position. Uses either Signal-/Slot-communication or invokes getPos blocking.
 
**void guiChangedSmallStep(double value)**

 Internal slot if the small step size is changed within the GUI (spinbox)
 

**void guiChangedLargeStep(double value)**

 Internal slot if the large step size is changed within the GUI (spinbox)
 

Signals
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**void RequestStatusAndPosition(bool sendActPosition, bool sendTargetPos)**

 Internal signal for c++-Code connected to the corresponding slot of the actuator.