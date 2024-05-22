.. include:: ../include/global.inc

.. |star| unicode:: U+002A

.. _install-all-in-one-dev-setup_creation:

All-In-One development creation
*******************************

This chapter here shows how the all-in-one development setup is created. In this
setup all required components for the |itom| development environment are installed/compiled.
The following steps must be executed in the given procedure.
In the chapter about :ref:`install_all-in-one_development_setup <install-all-in-one-dev-setup>`
you find how the development environment is installed.


Download Packages
==============================

Download all required software/packages for 64-bit version/ 32-bit version
---------------------------------------------------------------------------

All these steps here are shown for a 64-bit version of the all-in-one development setup.
A 32-bit version is created in the same way with the corresponding 32-bit software/packages.

First download all the needed software and packages. This download links are for
the current available software versions.

* `MS Visual Studio Community Edition <http://www.visualstudio.com/de-de/downloads/download-visual-studio-vs.aspx>`_
* `Qt Visual Studio Add-in <https://download.qt.io/development_releases/vsaddin/>`_
* `Qt for Windows x64 Online Installer <https://www.qt.io/download-qt-installer?hsCtaTracking=99d9dd4f-5681-48d2-b096-470725510d34%7C074ddad0-fdef-4e53-8aa8-5e8a876d6ab4>`_
* `CMake <http://cmake.org/download/>`_
* `Python Windows x64 executable installer <http://www.python.org/downloads/windows/>`_
* `Git x64 <https://git-scm.com/downloads>`_
* `TortoiseGit <http://tortoisegit.org/download/>`_
* `Doxygen <https://sourceforge.net/projects/doxygen/files/rel-1.8.11/>`_

Download packages for the 3rdPartyPCL tools
--------------------------------------------

* `OpenCV source <https://opencv.org/releases.html>`_
* `Boost binaries <https://www.boost.org/>`_
* `Eigen repository <https://eigen.tuxfamily.org/index.php?title=Main_Page>`_
* `VTK source <http://www.vtk.org/download/>`_
* `PCL source <http://github.com/PointCloudLibrary/pcl/>`_
* FLANN and QHULL can be copied from the current all-in-one development version.
* FLANN: repository <https://github.com/mariusmuja/flann/>
* `QHull repository <http://www.qhull.org/download/>`_


Compile/setup dependencies
==========================

Install Visual Studio
---------------------

Install Visual Studio Community Edition without optional features.

Creating the _install_ and 3rdParty folder
---------------------------------------------

Create a path on your hard drive with a long, long path name called **${MAINDIR}** (*only for QT5*).
Later, the all-in-one path on the destination computer must be shorter than this path
name, due to the Qt patching. For example your path can be called:
E:\\itom_all-in-one_development\\itom_all-in-one_development\\vs2017_qt5.12.1_x64\\.
Than create the following paths relative to the **${MAINDIR}**. 3rdPartyPCL is not
necessary to create a working |itom|.

.. figure:: images/all-in-one-create/folder_structure.png
    :scale: 100%
    :align: center

_install_ folder
``````````````````````````````````````

The _install_ folder is used to execute the installation procedure of the all-in-one
development. Copy the following files into **${MAINDIR}**/_install_ folder

.. figure:: images/all-in-one-create/folder_install.png
   :scale: 100%
   :align: center

optional folder
>>>>>>>>>>>>>>>>

This folder is for optional software, which is not necessary to build and compile
a working |itom|. This software is required further for development purposes.
Copy the following files into **${MAINDIR}**/_install_/optional folder

.. figure:: images/all-in-one-create/folder_optional.png
   :scale: 100%
   :align: center

qpatch folder
>>>>>>>>>>>>>>>>

The folder *qpatch* contains the files, which are needed to patch the prebuild
version of Qt version. Copy following files into **${MAINDIR}**/_install_/qpatch
folder. Change the **root** Qt path and the target path in the **create_files_to_patch.py**
file. Execute the script and check if the filenames in the **files-to-patch-windows** has been found.

.. figure:: images/all-in-one-create/folder_qpatch.png
   :scale: 100%
   :align: center



3rdParty folder
```````````````

Python folder
>>>>>>>>>>>>>

For the compilation of |itom|, it is not necessary to have a installed Python on
the computer. For the Python 3rdParty folder, first:

* Install the current version of **Python** on your computer.
* Copy the installed **Python** folder into the **${MAINDIR}**/3rdParty/Python folder.
* Deinstall it again.

.. warning::

    You have to copy the installed folder. Renaming and deinstall it does not work!

* Install numpy using pip

.. code-block:: bash

    python.exe -m pip install numpy


Qtx.xx.x folder
>>>>>>>>>>>>>>>

copy the output of the qt compilation process into this folder.
:ref:`build_dependencies_qt`


OpenCVx.x.x folder
>>>>>>>>>>>>>>>>>>

copy the output of the compilation process to this folder.
:ref:`build_dependencies_opencv`


3rdPartyPCL folder
``````````````````

For the 3rdPartyPCL folder the software packages Boost, Eigen, Flann and QHull
can be downloaded as binaries. VTK and PCL must be compiled on your computer.
Have the version dotted in the Name.

boostX.XX.X
>>>>>>>>>>>

Go for the prebuilt binaries(downloaded, see above)
Execute the **boost_1_69_0-msvc-12.0-64.exe** file and install boost on your hard
drive in a folder with a short path that is different from QT. Copy than the
folders **boost** and **lib** into **${MAINDIR}/3rdPartyPCL/boost1.69.0-64**.


EigenX.X.X
>>>>>>>>>>

Unzip from the Eigen zip-file the folders **Eigen** and **unupported** into
the **${MAINDIR}/3rdPartyPCL/Eigen3.3.7/**. (this folder)
for further information check with
http://eigen.tuxfamily.org/index.php?title=Main_Page

flannX.X.X
>>>>>>>>>>

Copy the downloaded flannX.X.X into **${MAINDIR}/3rdPartyPCL/flann1.9.1** (here).
For further info: https://github.com/mariusmuja/flann
or check with
:ref:`build_dependencies_flann_qhull`


QHullxxxx.x
>>>>>>>>>>>

Copy qhull-2015.2 into **${MAINDIR}/3rdPartyPCL/qhull-2015.2**.
or get some version from http://www.qhull.org/download/
or check with
:ref:`build_dependencies_flann_qhull`

VTKx.x.x
>>>>>>>>

Copy the output of the compilation process to this folder
:ref:`build_dependencies_vtk`


PCLx.x.x
>>>>>>>>

Copy the output of the compilation process to this folder
:ref:`build_dependencies_pcl`


Modify setupscript
=============================================

Update the version of the python packages (pip, setuptools, wheel) of the **setup_itom_all-in-one.bat**.

The setup.py file needs some changes to work with the new version of the
software/packages. The setup checks, if the packages are given in the right version.

First set the following variables in the beginning of the file.

* Set the **qtOriginalBuildPath**. E. g.: "C:\\itom_all-in-one_development\\itom_all-in-one_development\\itom_all-in-one_development\\VS2017_Qt5.12.1_x64\\3rdParty\\Qt5.12.1\\5.12.1\\msvc2017_64"
* Set the **qtNewBuildPath**. E. g.: "../3rdParty/Qt5.12.1/5.12/msvc2017_64"
* Set the **numpyRequiredVersion** to the Numpy version, which is attached to the all-in-one development setup. E. g.: "1.11.0"
* Set the **pythonRequiredVersion** to the python version, which is attached to the all-in-one development setup. E. g.: "3.5."

Check in the function **generateCMakeDict** the version of Visual Studio and the
paths of **CMake, OpenCV, Python library version, VTK, PCL, Eigen, Flann, QHull**.

.. figure:: images/all-in-one-create/Setup_packages_versions.PNG
   :width: 100%
   :align: center

Check in the function **installNumpy**, if the numpy whl file names are right.

.. figure:: images/all-in-one-create/Setup_numpy_file.PNG
   :width: 631px
   :align: center

Check in the function **upgradePip**, **installPyflakes**, **installJedi** the filenames of the whl-files.

Pack everything together
=============================================
use 7zip and rar format to pack windows compiled binaries together to upload them to
sourceforge
