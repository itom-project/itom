.. include:: /include/global.inc

Python scripting language
#########################

One main component of |itom| is the integrated scripting language |python|. In order to make |itom| ready for the future, |python| is based on the version-family 3.
If you already know |python| or already have used |python|, you probabily know about the huge functionality which already is provided by |python| itself or by one of its
various modules, that are available in the internet. The good is, that you can execute the same |python|-scripts in |itom|. There are only few limitations concerning
some |python| modules (see :ref:`script-python-limitations`).

The main purpose of the embedded |python| interpreter in |itom| is to access the specific functionalities provided by |itom|. This is done by the |python|-module **itom**, that
is importable only in the embedded |python| interpreter in |itom|. This module includes interfaces for hardware and algorithm plugins of |itom| as well as classes that wrap the
most important internal data structures of |itom|, like matrices (class **dataObject**), point clouds (class **pointCloud**) or polygon meshes (class **polygonMesh**). Additionally
the module provides functions to manipulate or extend the graphical user interface of |itom| as well as to create own dialogs or windows (provided by the class **ui** and **uiItem**).

Beside the module **itom**, it is recommended to use the functionalities provided by the |python| packages **Numpy**, **Scipy** and **Matplotlib**. During the development of |itom| a
high compatibility especially to those modules has been taken into account. For instance it is possible to render the **Matplotlib** figures in user defined windows, created by the class **ui** of the module **itom**. Additionally, the **Numpy** array is compatible to the |itom| internal **dataObject** or **npDataObject**.

The function and class reference for the itom-module can be found under :ref:`itom-Script-Reference`.

If you don't know |python|, you probably should have a look at our python tutorial page that also contains links to recommended tutorials in the internet:

.. toctree::
    :maxdepth: 1
    
    py-tutorial/pytut_about.rst
    
An introduction to the functionalities provided by the module **itom** are found under:

.. toctree::
    :maxdepth: 1
    
    pymod-itom.rst



.. toctree::
    :maxdepth: 1

    script-execution.rst
    
    pymod-itom.rst
    pymod-scipy.rst
    pymod-matplotlib.rst
    pymod-numpy.rst
    pymod-limitations.rst
    glossary.rst


