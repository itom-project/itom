Save and load images and other files
**************************************

Native file formats
=============================

itom has a native support to load and save various file formats. Additionally, algorithm plugins can provide further filters to load and save more file formats.
All supported formats (native and plugin-based) are considered in the GUI such that you can open a file using the *file >> open* menu, by double-clicking a file in the file system dialog or by using the import / export buttons in the workspace.

The following formats are natively supported:

* **idc**: This is the default itom file format (itom data collection) and is able to store entire Python structures (e.g. dictionaries, lists, tuples...) containing data objects, point clouds or other python objects. This file format is written and read using the module :py:mod:`pickle` from python.
* **mat**: Similar to **idc** you can import or export entire data structures from and to Matlab. This is only available if the package **scipy** is installed.
* **ido**, **idh**: This is a xml-based data format for single data objects or only meta and header information of a single data object.

Here are some examples about the natively supported. At first, let us create some exemplary data objects:

.. code-block:: python
    
    #randomly filled matrix of size 100x100, type: uint8
    obj1 = dataObject.randN([100,100],'uint8')
    
    #50x100 matrix filled with float32 values
    obj2 = dataObject([50,100], 'float32')
    obj2[0:25,:] = 0.0
    obj2[25:50,:] = 10.3
    obj2[25:50,50:100] = -5.25
    
    #768x1024 coloured data object (type: rgba32), background: white,
    #left side: transparent, in the middle three horizontal bars in red,
    #green and blue.
    obj3 = dataObject([768,1024],'rgba32')
    obj3[:,0:512] = rgba32(255,255,255,0) #transparent (alpha=0)
    obj3[:,512:1024] = rgba32(255,255,255,255)
    obj3[100:300,200:800] = rgba32(255,0,0,255) #red
    obj3[300:500,200:800] = rgba32(0,255,0,255) #green
    obj3[500:700,200:800] = rgba32(0,0,255,255) #blue
        
Now, we want to save all these three data objects together into one **idc** file using the method :py:meth:`itom.saveIDC`. 
**idc** files always contain a dictionary where you can save whatever you want to (not only data objects). When loading 
an **idc** file (by :py:meth:`itom.loadIDC`, you get the original dictionary back:

.. code-block:: python
    
    # save idc file
    saveIDC("C:/test.idc", {"mat1":obj1, "mat2":obj2, "mat3":obj3})
    # remember: if you use \ in pathes, replace them by \\
    
    # load the file again
    myDict = loadIDC("C:/test.idc")
    obj1new = myDict["mat1"]
    obj2new = myDict["mat2"]

.. note::
    
    In these examples, the methods and classes from the :py:mod:`itom` are written without the module name as prefix.
    These is possible, since the :py:mod:`itom` is globally imported at startup of itom. However this holds only for the global
    workspace. 
    
If you wish to save the same objects to a Matlab **mat** file, this is also possible via dictionaries. When loaded in Matlab,
each item in the dictionary is a variable in the workspace whose name is the key of the item. The save and load methods are 
:py:meth:`itom.saveMatlabMat` and :py:meth:`itom.loadMatlabMat`:

.. code-block:: python
    
    # save matlab file
    saveMatlabMat("C:/test.mat", {"mat1":obj1, "mat2":obj2, "mat3":obj3})
    
    # load the file again
    myDict = loadMatlabMat("C:/test.mat")
    obj1new = myDict["mat1"]
    obj2new = myDict["mat2"]
    
If a data object is saved in a Matlab **mat** file, Matlab will load this data object as cell array that contains both the matrix data
itself and all meta information (scaling, offset, tags, ...).

If you want to export single data objects in a readable format, use the methods :py:meth:`itom.saveDataObject` and :py:meth:`itom.loadDataObject`.
Both export or import into / from the xml-based files **ido** (entire data object with data and meta information) and **idh**
(only meta information (header) of data object). In the first format, header information is directly readable in the file while the matrix
data is encoded in a base64 format.

Plugin-based file formats
=============================

Plugins can provide filters for saving or loading the following objects:

* data objects
* point clouds
* polygon meshes

If any filter indicates to support the corresponding file input or file output interface, this filter is automatically recognized and integrated in the GUI. Nevertheless, these filters can be called like any other filter in |itom|.

Most filters for loading any image formats are included in the plugin **dataObjectIO**. The filter documentation of this plugin gives detailed information
about every single filter. Loading or saving point clouds or polygonal meshes are included in the plugin **PclTools**.

Image file formats
=============================

As mentioned in the section above, plugins can provide filters to save or load data objects.
The plugin **dataObjectIO** contains many filters to save into common image formats and load them back to data objects. Click **info** in the context
menu of any algorithm filter to get more information about this filter.

All image-based file filters follow these rules how to handle different data types:

* uint8 or uint16 are saved as gray-values (8bit or if supported as 16bit) or if the image format allows color are saved according to the defined color palette.
* float32 or float64 are saved as gray-values (8bit or if suppored as 16bit) or according to the defined color palette. Therefore the values must be between 0.0 and 1.0.  Values outside these borders are clipped. If the image format supports RGBA, invalid values are saved as transparent values (alpha=zero) else as black values.
* rgba32 can be saved as 'rgb' (full opacity), 'rgba' (alpha channel is considered, not supported by all formats) or gray formats, where the color image is transformed to gray. if a format from a color palette is indicated, the color image is transformed to gray first and then interpreted using the indicated color palette.

Among others, the following color formats are supported: bmp, jpg, png, gif (read-only), tiff, xpm, xbm, ras, pgm, ppm...

Loading these files can mainly be achieved by the filter **loadAnyImage**:

.. code-block:: python
    
    reload_tiff_rgba=dataObject()
    filter("loadAnyImage",reload_tiff_rgba, 'pic_rgba.tiff','asIs')
    
'asIs' means that the data is loaded without further transformations (if possible), hence, a color data format is loaded to a rgba32 data object, a uint8 gray image is loaded to uint8 and so on. However, you can also choose that you want the image to be always converted to gray, you can choose a specific color channel...

For saving to different color formats, there is usually a specific filter for each format. This allows passing further individual parameters like the color map for *png*. This is indicates if fixed- or floating-point data objects should be interpreted with a specific color map. The output is then a color image instead of a gray one:

.. code-block:: python
    
    filter("savePNG", obj1, 'C:/pic_falseColor.png', 'hotIron')

For more examples about saving and loading data, see the demo file **demoLoadSaveDataObjects.py** in the demo-folder.