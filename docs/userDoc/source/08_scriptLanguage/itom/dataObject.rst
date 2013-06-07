.. include:: ../../include/global.inc

.. _itomDataObject:

DataObject 
==========

.. moduleauthor:: ITO
.. sectionauthor:: ITO

Introduction
------------

In |itom|, the class :py:class:`~itom.dataObject` is the main array object. Arrays in |itom| can have the following properties:

* unlimited number of dimensions
* each dimension can have an arbitrary size
* possible data types:
    .. code-block:: python
        
        "uint8"      #unsigned integer, 8 bit [0,255]
        "int8"       #signed integer, 8 bit [-128,127]
        "uint16"     #unsigned integer, 16 bit [0,65536]
        "int16"      #signed integer, 16 bit [-32768,32767]
        "uint32"     #unsigned integer, 32 bit
        "int32"      #signed integer, 32 bit
        "float32"    #floating point, 32 bit single precision
        "float64"    #floating point, 64 bit double precision
        "complex64"  #complex number with two float32 components
        "complex128" #complex number with two float64 components

Before giving a short tutorial about how to use the class :py:class:`~itom.dataObject`, the base idea and concept of the array structure should be explained. If you already now the huge |python| module **Numpy** with its base array class **numpy.array**, one will ask why another similar array class is provided by |itom|. The reasons for this are as follows:

* The python class :py:class:`~itom.dataObject` is just a wrapper for the |itom| internal class **DataObject**, written in C++. This array structure is used all over |itom| and also passed to any plugin instances of |itom|. Internally, the C++ class **DataObject** is based on OpenCV-matrices, such that functionalities provided by the open-source Computer-Vision Library (OpenCV) can be used by |itom|.
* The class **dataObject** should also be used to store real measurement data. Therefore it is possible to add tags and other meta information to every dataObject (like axes descriptions, scale and offset values, protocol entries...).
* Usually, array classes (like the class **Numpy.array**) store the whole matrix in one non-interrupting block in memory. Due to the working principle of every operating system, it is sometimes difficult to allocate a huge block in memory. Therefore, **dataObject** only stores the sub-matrices of the last two-dimensions in single blocks in memory, while the first **n-2** dimensions of the array are represented by one vector in memory, where every cell is pointing to the corresponding sub-matrix (called plane). Using this concept, huger arrays can be allocated without causing a memory error.

Creating a dataObject
---------------------

In general, a :py:class:`~itom.dataObject` is created like any other class instance in |python|, hence the constructor of class :py:class:`~itom.dataObject` is called. For a full reference of the constructor of class **dataObject**, type

.. code-block:: python
    
    help(dataObject)

In the following example, some dataObjects of different size and types are created. Using these constructors, the content of the created array is arbitrary at initialization:

.. code-block:: python
    :linenos:
    
    #1. empty dataObject, dimensions: 0, size: []
    a = dataObject()
    
    #2. one dimensional dataObject
    #  a one dimensional dataObject already is
    #  allocated as an array of size [1 x n]
    b = dataObject([5], "float32") #size [1x5]
    
    #3. 5 x 3 array, type: int8
    c = dataObject([5,3], "int8")
    
    #4. 2 x 5 x 10 array, type: complex128
    #  here two planes of size [5x10] are created and a vector with two items points to them
    d = dataObject([2,5,10], "complex128")
    
    #5. 2 x 5 x 10 array, type: complex128, continuous
    #  This matrix has the same size and type than matrix
    #  'd' above. However, the continuous keyword indicates,
    #  that python should already allocate all planes in
    #  one block. Then the data object can be converted in
    #  a numpy.array without the need of copying the data block
    #  in memory. It is useful to use this keyword, if you
    #  often want to switch between dataObject and numpy.arrays.
    #  However consider that this is not recommended for huge
    #  matrices.
    e = dataObject([2,5,10], "complex128", continuous = True)

You can also use the copy constructor of class **dataObject** in order to create
a dataObject from another array-like object or a sequence of numbers (tuple, list...).
In |python| it is usual, that different objects share their memory (for arrays the memory
is mainly the data block(s)) as long as possible, such that memory and execution time is saved. This is also the case when using the copy constructor. See the **Numpy** documentation for more information about this. The main thing you should know is, that if you change the value of any cell of an array, the corresponding value is also changed in all arrays, that share their memory with the dataObject.

.. code-block:: python
    :linenos:
    
    #1. create dataObject from any array-like object (e.g. Numpy array)
    import numpy as np
    a = np.ndarray([5,7])
    b = dataObject(a) #b has the continuous flag set
    
    #2. create dataObject from a tuple of values
    #  any object, that python can interpret as sequence can be used
    #  in order to initialize the data object. The dataObject can have
    #  an arbitrary size or number of dimensions, if the total number
    #  of elements fits to the length of the given input sequence.
    #  In this case, the sequence is totally copied into the data object.
    #  The values are filled row-by-row into the array, also called as
    #  c-continuous creation.
    c = (2,7,4,3,8,9,6,2) #8 values
    d = dataObject([2,4], data = c)
    
    #3. create a dataObject as shallow copy of another dataObject
    e = dataObject(d)
    
    
    
    
    

Some text about the basic idea of the dataObject and how it works.
Some additional pictures. 

For a detailed methods-summery of the *dataObject* see :ref:`ITOM-Script-Reference`.