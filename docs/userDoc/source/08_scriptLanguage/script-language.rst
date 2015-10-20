.. include:: /include/global.inc

.. _scriptLanguage:

Python scripting language
#########################

Introduction
*************

One main component of |itom| is the integrated scripting language |python|. In order to make |itom| ready for the future, |python| is based on the version-family 3.
If you already know |python| or already have used |python|, you probably know about the huge functionality which already is provided by |python| itself or by one of its
various modules, that are available in the internet. The good is, that you can execute the same |python|-scripts in |itom|. There are only few limitations concerning
some |python| modules (see :ref:`script-python-limitations`).

If you are a |python| beginner, you probably should have a look at our tutorial page that also contains links to recommended tutorials in the internet:

.. toctree::
    :maxdepth: 1
    
    py-tutorial/pytut-about.rst

If you use |python| in |itom| the following documents might give you more information about limitations, automatically reloading modules...:

.. toctree::
    :maxdepth: 1
    
    pymod-limitations.rst
    pymod-problems-and-solutions.rst
    pymod-reload-modules.rst

Python module itom
********************

The main purpose of the embedded |python| interpreter in |itom| is to access the specific functionalities provided by |itom|. This is done by the |python|-module :py:mod:`itom`, that
is importable only in the embedded |python| interpreter in |itom|. This module includes interfaces for hardware and algorithm plugins of |itom| as well as classes that wrap the
most important internal data structures of |itom|, like matrices (class :py:class:`~itom.dataObject`), point clouds (class :py:class:`~itom.pointCloud`) or polygon meshes (class :py:class:`~itom.polygonMesh`). Additionally
the module provides functions to manipulate or extend the graphical user interface of |itom| as well as to create own dialogs or windows (provided by the class :py:class:`~itom.ui` and :py:class:`~itom.uiItem`).

More information about the module :py:mod:`itom` can be found under:

.. toctree::
    :maxdepth: 3
    
    pymod-itom.rst
    pymod-load-save-files.rst
    glossary.rst

The full script reference of the module :py:mod:`itom` can be found under :ref:`itom-Script-Reference`.
    
Further python packages
************************** 

Beside the module :py:mod:`itom`, it is recommended to use the functionalities provided by the |python| packages **Numpy**, **Scipy** and **Matplotlib**. During the development of |itom| a
high compatibility especially to those modules has been taken into account. For instance it is possible to render the **Matplotlib** figures in user defined windows, created by the class :py:class:`~itom.ui` of the module :py:mod:`itom` (see :ref:`qtdesigner`). Additionally, the **Numpy** array is compatible to the |itom| internal :py:class:`~itom.dataObject` or :py:class:`~itom.npDataObject`.

.. toctree::
    :maxdepth: 1

    pymod-numpy.rst
    pymod-scipy.rst
    pymod-matplotlib.rst
    
Other recommended packages are:

* **scikit-image** for image processing
* **PIL** is the python image library
* **sphinx** for creating this documentation

See :ref:`python package manager <gui-pipmanager>` for more information about getting packages.


Tutorials, documentations about Python 3
*****************************************

See the following sources for more information, tutorials and documentations about Python 3 and Numpy:

* `Python-Kurs (v3) <http://www.python-kurs.eu/python3_kurs.php>`_ (German)
* `Python Course (v3) <http://www.python-course.eu/python3_course.php>`_ (English)
* `Official Python 3 documentation <http://docs.python.org/3/>`_
* `Moving from Python 2 to Python 3 (Cheatsheet) <http://ptgmedia.pearsoncmg.com/imprint_downloads/informit/promotions/python/python2python3.pdf>`_
* `Dive into Python (v3) <http://www.diveinto.org/python3/>`_
* `Numpy for Matlab users <http://wiki.scipy.org/NumPy_for_Matlab_Users>`_
    