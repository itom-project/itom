.. include:: ../../include/global.inc

.. sectionauthor:: Marc Gronle, ITO

.. _plugin-Algo-CatInterfaces:

Algorithm Categories and Interfaces
===================================

You can assign to every filter- or widget-method a certain category and/or algorithm interface, the method belongs or fits to. This is useful such that |itom| can ...

* categorize your filter or widget.
* is able to integrate the filter in specific parts of the GUI, e.g. the file load or save process or dialog.
* might use your filters in image processing chains.
* ...

Algorithm Categories
--------------------


Algorithm Interfaces
--------------------

A programmer who wants to implement a filter or widget-method that fits to a specific algorithm interface can be forced by the interface to consider a certain set of rules. These rules can be:

* The first **n** mandatory parameters can be determined. For each of these parameters the type and optionally some constraints by adding a meta information instance to the parameter are given.
* The first **m** output parameter can be determined. Type and meta information (optional) are given as well).
* The maximum number of mandatory parameters is determined, but can also be set to infinity (INT_MAX).
* The maximum number of optional parameters is determined, but can also be set to infinity (INT_MAX).
* The maximum number of output parameters is determined, but can also be set to infinity (INT_MAX).
* The syntax or general form of the meta information string, added to the classes **FilterDef** or **AlgoWidgetDef** can be given.

If a filter or widget-method pretends to implement a certain interface, |itom| checks at startup if the constraints are fulfilled are if this is not the case, the filter or widget-method is rejected and cannot be used within |itom|. The reason for the rejection can be seen by open the dialog **loaded plugins...** within the **help**-menu of |itom|.

The following interfaces are available (in the enumeration **tAlgoInterface**, member of class **ito::AddInAlgo**:

iNotSpecified
+++++++++++++

This is the default implementation and indicates that your filter or widget-method does not fit to any interface.

iReadDataObject
+++++++++++++++

Filters fitting to this interface provide the functionality to read a certain file and load its content to an instance of dataObject.

**Argument limitations**

+----------------------+-------------------------------------------+
| Parameter group      | Max. number of parameters                 |
+======================+===========================================+
| mandatory parameters | infinity                                  |
+----------------------+-------------------------------------------+
| optional parameters  | infinity                                  |
+----------------------+-------------------------------------------+
| output parameters    | 0                                         |
+----------------------+-------------------------------------------+

**Mandatory parameters**

+--------------------------+----------------------+----------------------------------------------------------+--------------------------------------------------+
| Parameters               | Type                 | Description                                              | Further limitations (meta information)           |
+==========================+======================+==========================================================+==================================================+
| #1 dataObject            | DObjPtr, In, Out     | DataObject, where the file is loaded to                  | None                                             |
+--------------------------+----------------------+----------------------------------------------------------+--------------------------------------------------+
| #2 filename              | String, In           | absolute filename of the file to load                    | None                                             |
+--------------------------+----------------------+----------------------------------------------------------+--------------------------------------------------+

**Output parameters**

- No -

**Meta information string**

This string must contain the file-filters with file-endings that this filter is able to load. Different file filters are separated by a double-semicolon (;;). Each
filter begins with its name (arbitrary string), followed by a space and a sequence of file-endings within a pair of brackets. Examples are:

* Images (\*.bmp \*.png \*.jpg \*.pgm)
* Bitmap (\*.bmp);;JPEG (\*.jpg)
* Text-Files (\*.txt)

iWriteDataObject
++++++++++++++++

Filters fitting to this interface provide the functionality to export a dataObject with a specific file format.

**Argument limitations**

+----------------------+-------------------------------------------+
| Parameter group      | Max. number of parameters                 |
+======================+===========================================+
| mandatory parameters | infinity                                  |
+----------------------+-------------------------------------------+
| optional parameters  | infinity                                  |
+----------------------+-------------------------------------------+
| output parameters    | 0                                         |
+----------------------+-------------------------------------------+

**Mandatory parameters**

+--------------------------+----------------------+----------------------------------------------------------+--------------------------------------------------+
| Parameters               | Type                 | Description                                              | Further limitations (meta information)           |
+==========================+======================+==========================================================+==================================================+
| #1 dataObject            | DObjPtr, In          | DataObject that should be exported                       | None                                             |
+--------------------------+----------------------+----------------------------------------------------------+--------------------------------------------------+
| #2 filename              | String, In           | absolute filename of the file                            | None                                             |
+--------------------------+----------------------+----------------------------------------------------------+--------------------------------------------------+

**Output parameters**

- No -

**Meta information string**

This string must contain the file-filters with file-endings that this filter is able to load. Different file filters are separated by a double-semicolon (;;). Each
filter begins with its name (arbitrary string), followed by a space and a sequence of file-endings within a pair of brackets. Examples are:

* Images (\*.bmp \*.png \*.jpg \*.pgm)
* Bitmap (\*.bmp);;JPEG (\*.jpg)
* Text-Files (\*.txt)

iReadPointCloud
+++++++++++++++

Filters fitting to this interface provide the functionality to read a certain file and load its content to an instance of pointCloud.

**Argument limitations**

+----------------------+-------------------------------------------+
| Parameter group      | Max. number of parameters                 |
+======================+===========================================+
| mandatory parameters | infinity                                  |
+----------------------+-------------------------------------------+
| optional parameters  | infinity                                  |
+----------------------+-------------------------------------------+
| output parameters    | 0                                         |
+----------------------+-------------------------------------------+

**Mandatory parameters**

+--------------------------+------------------------+----------------------------------------------------------+--------------------------------------------------+
| Parameters               | Type                   | Description                                              | Further limitations (meta information)           |
+==========================+========================+==========================================================+==================================================+
| #1 pointCloud            | PointCloudPtr, In, Out | PointCloud, where the file is loaded to                  | None                                             |
+--------------------------+------------------------+----------------------------------------------------------+--------------------------------------------------+
| #2 filename              | String, In             | absolute filename of the file to load                    | None                                             |
+--------------------------+------------------------+----------------------------------------------------------+--------------------------------------------------+

**Output parameters**

- No -

**Meta information string**

This string must contain the file-filters with file-endings that this filter is able to load. Different file filters are separated by a double-semicolon (;;). Each
filter begins with its name (arbitrary string), followed by a space and a sequence of file-endings within a pair of brackets. Examples are:

* Images (\*.bmp \*.png \*.jpg \*.pgm)
* Bitmap (\*.bmp);;JPEG (\*.jpg)
* Text-Files (\*.txt)

iWritePointCloud
++++++++++++++++

Filters fitting to this interface provide the functionality to export a pointCloud with a specific file format.

**Argument limitations**

+----------------------+-------------------------------------------+
| Parameter group      | Max. number of parameters                 |
+======================+===========================================+
| mandatory parameters | infinity                                  |
+----------------------+-------------------------------------------+
| optional parameters  | infinity                                  |
+----------------------+-------------------------------------------+
| output parameters    | 0                                         |
+----------------------+-------------------------------------------+

**Mandatory parameters**

+--------------------------+----------------------+----------------------------------------------------------+--------------------------------------------------+
| Parameters               | Type                 | Description                                              | Further limitations (meta information)           |
+==========================+======================+==========================================================+==================================================+
| #1 pointCloud            | PointCloudPtr, In    | PointCloud that should be exported                       | None                                             |
+--------------------------+----------------------+----------------------------------------------------------+--------------------------------------------------+
| #2 filename              | String, In           | absolute filename of the file                            | None                                             |
+--------------------------+----------------------+----------------------------------------------------------+--------------------------------------------------+

**Output parameters**

- No -

**Meta information string**

This string must contain the file-filters with file-endings that this filter is able to load. Different file filters are separated by a double-semicolon (;;). Each
filter begins with its name (arbitrary string), followed by a space and a sequence of file-endings within a pair of brackets. Examples are:

* Images (\*.bmp \*.png \*.jpg \*.pgm)
* Bitmap (\*.bmp);;JPEG (\*.jpg)
* Text-Files (\*.txt)

iReadPolygonMesh
++++++++++++++++

Filters fitting to this interface provide the functionality to read a certain file and load its content to an instance of polygonMesh.

**Argument limitations**

+----------------------+-------------------------------------------+
| Parameter group      | Max. number of parameters                 |
+======================+===========================================+
| mandatory parameters | infinity                                  |
+----------------------+-------------------------------------------+
| optional parameters  | infinity                                  |
+----------------------+-------------------------------------------+
| output parameters    | 0                                         |
+----------------------+-------------------------------------------+

**Mandatory parameters**

+--------------------------+------------------------+----------------------------------------------------------+--------------------------------------------------+
| Parameters               | Type                   | Description                                              | Further limitations (meta information)           |
+==========================+========================+==========================================================+==================================================+
| #1 polygonMesh           | PolygonMeshPtr, In, Out| PolygonMesh, where the file is loaded to                 | None                                             |
+--------------------------+------------------------+----------------------------------------------------------+--------------------------------------------------+
| #2 filename              | String, In             | absolute filename of the file to load                    | None                                             |
+--------------------------+------------------------+----------------------------------------------------------+--------------------------------------------------+

**Output parameters**

- No -

**Meta information string**

This string must contain the file-filters with file-endings that this filter is able to load. Different file filters are separated by a double-semicolon (;;). Each
filter begins with its name (arbitrary string), followed by a space and a sequence of file-endings within a pair of brackets. Examples are:

* Images (\*.bmp \*.png \*.jpg \*.pgm)
* Bitmap (\*.bmp);;JPEG (\*.jpg)
* Text-Files (\*.txt)

iWritePolygonMesh
+++++++++++++++++

Filters fitting to this interface provide the functionality to export a polygonMesh with a specific file format.

**Argument limitations**

+----------------------+-------------------------------------------+
| Parameter group      | Max. number of parameters                 |
+======================+===========================================+
| mandatory parameters | infinity                                  |
+----------------------+-------------------------------------------+
| optional parameters  | infinity                                  |
+----------------------+-------------------------------------------+
| output parameters    | 0                                         |
+----------------------+-------------------------------------------+

**Mandatory parameters**

+--------------------------+----------------------+----------------------------------------------------------+--------------------------------------------------+
| Parameters               | Type                 | Description                                              | Further limitations (meta information)           |
+==========================+======================+==========================================================+==================================================+
| #1 polygonMesh           | PolygonMeshPtr, In   | PolygonMesh that should be exported                      | None                                             |
+--------------------------+----------------------+----------------------------------------------------------+--------------------------------------------------+
| #2 filename              | String, In           | absolute filename of the file                            | None                                             |
+--------------------------+----------------------+----------------------------------------------------------+--------------------------------------------------+

**Output parameters**

- No -

**Meta information string**

This string must contain the file-filters with file-endings that this filter is able to load. Different file filters are separated by a double-semicolon (;;). Each
filter begins with its name (arbitrary string), followed by a space and a sequence of file-endings within a pair of brackets. Examples are:

* Images (\*.bmp \*.png \*.jpg \*.pgm)
* Bitmap (\*.bmp);;JPEG (\*.jpg)
* Text-Files (\*.txt)


Definition of new algorithm interfaces
--------------------------------------

At first the categories and interfaces are determined as an enumeration value in the enumerations **ito::AddInAlgo::tAlgoCategory** and **ito::AddInAlgo::tAlgoInterface** (file **addInInterface.h**).

The constraints and rules for every interface are implemented in the methods **init** and **getTags** of class **AlgoInterfaceValidator**. Besides the syntax of the meta information string all necessary information is given in the method **init**.
