Load and save images and other files
**************************************

itom has a native support to load and save various file formats. Additionally, plugins can provide further filters to load and save more file formats.
All supported formats (native and plugin-based) are considered in the GUI such that you can open a file using the file>>open button, by double-clicking on
the file in the file system dialog or by using the import / export buttons in the workspace.

The following formats are natively supported:

* **idc**: This is the default itom file format (itom data collection) and is able to store entire Python structures (e.g. dictionaries, lists, tuples...) containing data objects, point clouds or other python objects. This file format is written and read using the module *pickle* from python.
* **mat**: Similar to **idc** you can import or export entire data structures from and to Matlab. This is only available if the package **scipy** is installed.
* **ido**: This is a xml-based data format for single data objects only.

Plugins can provide filters for saving or loading the following objects:

* data objects
* point clouds
* polygon meshes

If any filter indicates to support the corresponding file input or file output interface, this filter is automatically recognized and integrated in the GUI. Nevertheless, these filters can be called like any other filter in |itom|