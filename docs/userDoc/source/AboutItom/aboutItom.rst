.. include:: ../include/global.inc

About ITOM
****************

What is |itom|?
===================

|itom| is the new measurement software of the ito. With this software we wanted to create a new fast system with simple and unitary integration of hardware. It should be a versatile tool for building the control software for any measurement setup with high adaptability to different situations. Therefore it is necessary to easily do modification in the software or transfer code blocks from one program to another. But not only that it is possible to control your hardware, you also can use |itom| for analysing or processing your acquired data. At the end we want to give you a software as replacement of Matlab, Labview,... and all with high process speed due to C++. Even though |itom| is a powerful software it is easy to use and so everyone can learn to handle it.

The following table shows how we could approach all our requirements:

+---------------------------------------------------------------+---------------------------+
| What we wanted                                                | How we did it             |
+===============================================================+===========================+
| fast (close to hardware) software                             | C++                       |
+---------------------------------------------------------------+---------------------------+
| modern multi-platform GUI                                     | Qt-Framework              |
+---------------------------------------------------------------+---------------------------+
| easy integration of different hardware (camera, actuator,...) | Plugin system             |
+---------------------------------------------------------------+---------------------------+
| fast, robust and easy to learn script language                | Python                    |
+---------------------------------------------------------------+---------------------------+

In the figure below you can see the three columns on which |itom| is based:

1. Python
2. Plugins
3. GUI

.. figure:: 3columns.png

Based on this three columns you can control measurement applications, basic setups or scripted image processing.

To lean more about how to control itom via script language or the GUI proceed with :ref:`gettingStarted`.



**TODO**: Add the basic idea and structure of ITOM and explain it in more detail than this above




Impressum
===================

| Institut fuer Technische Optik
| Universitaet Stuttgart
| Pfaffenwaldring 9
| 70569 Stuttgart
|
| **Bug report:**
| http://obelix/mantis/login_page.php

Licence
===================

Itom Licence
~~~~~~~~~~~~~~~~~~~~~

This help was build for iTOM-version |version| which correspond to SVN-Revision |release|
This itom-version will be licensed unter LGPL vX.X (probably 2.0, 2.1, 3.0).

Package Licences
~~~~~~~~~~~~~~~~~~~~~

Your main programm of itom uses the following third party packages:

- `openCV <http://opencv.willowgarage.com/wiki/>`_ by Willow Garage under BSD-license

- `point cload library <http://www.pointclouds.org/>`_ by Willow Garage under BSD-license

- The `QT-Framework <http://qt.nokia.com/>`_ by Nokia unter LGPL.

- `QwtPlot <http://qwt.sourceforge.net/>`_ by Uwe Rathmann and Josef Wilgen under LGPL with additional `exceptions <http://qwt.sourceforge.net/qwtlicense.html>`_.

- QPropertyEditor under LGPL

- `QScintilla <http://www.riverbankcomputing.co.uk/software/qscintilla/intro>`_ in version 2.6.0 by Riverbank Computed Limited under GPL with additional exceptions.

- QScintilla is a port to Qt of Neil Hodgson's `Scintilla <http://www.scintilla.org/>`_ C++ editor control.

- `Python <http://www.python.org/>`_ by Python Software Foundation unter Python-License (similar to BSD license).

- Python-package `NumPy <http://numpy.scipy.org/>`_ and `SciPy <http://www.scipy.org/>`_ under BSD compatible license.

- `MatPlotLib <http://matplotlib.sourceforge.net/>`_ under BSD compatible license.


Further used third party packages
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- Google test framework by Google under New BSD-license
    
- PythonQt, the python interaction with QT was inspired by PythonQt (under LGPL).

Some of the PlugIns may contain further third party packages, e.g.:

- NVidia CUDA SDK by NVidia under NVidia CUDA SDK License

- `OpenMP <http://openmp.org/wp/>`_

- Hardware dependent third party driver and SDKs with different license models,

or may be published under different terms of conditions than the main itom-programm. So please check plugin-license before distributing itom-plugins. 