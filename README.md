# itom

[![Generic badge](https://img.shields.io/badge/powered%20by-ITO-blue)](https://www.ito.uni-stuttgart.de/)
[![License: LGPL v3](https://img.shields.io/badge/License-LGPL_v3-blue.svg)](https://www.gnu.org/licenses/lgpl-3.0)

[![made-with-python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/)
[![made-with-opencv](https://img.shields.io/badge/Made%20by-OpenCV-green)](https://opencv.org/)
[![made-with-qt](https://img.shields.io/badge/Made%20by-Qt-brightgreen)](https://www.qt.io/product/framework)

Welcome to the open source software [``itom``](https://itom-project.github.io/ "``itom``"). In addition to a [Python](https://www.python.org/ "Python") IDE, it enables the operation of **measurement systems** with multiple **hardware components**, such as cameras, AD converters, actuators, motor stages, and the management of your laboratory automation. The **graphical user interface provides** a quick and easy access to all components, complex measurement tasks and algorithms can be scripted using the embedded python scripting language and self-defined user interfaces finally provide a possibility to adapt [``itom``](https://itom-project.github.io/ "``itom``")to your special needs. External hardware or algorithms are added to [``itom``](https://itom-project.github.io/ "``itom``") by an integrated plugin system.

In order to learn more about [``itom``](https://itom-project.github.io/ "``itom``"), see the official homepage [itom-project.github.io](https://itom-project.github.io/) or read the [user documentation](https://itom-project.github.io/latest/docs/index.html)

## What is this project for?

This project contains the core application of: 
* [itom](https://github.com/itom-project/itom) core application

Furthermore, you may also need the ``plugins`` and ``designer plugins`` repositories to get all itom functionalities.
* [plugins](https://github.com/itom-project/plugins) hardware/software plugins
* [designer plugins](https://github.com/itom-project/designerPlugins) widget plugins

## How do I get set up?

* In order to get itom either download the ready-to-use setups for Windows 32bit and 64bit. Use the [all-in-one installer](https://sourceforge.net/projects/itom/files/all-in-one-build-setup/ "all-in-one installer") in order to get itom including Python and some important Python packages or use the simple installer if you already have an appropriate version of Python installed on your computer.
* Clone the central repositoy [itomProject](https://github.com/itom-project/itomProject) with the **--recursive** Option.
```bash
git clone --recursive --remote git@github.com:itom-project/itomProject.git
```
This will download the submodule repositories [itom](https://github.com/itom-project/itom) core, plugins [plugins](https://github.com/itom-project/plugins) and [designer plugins](https://github.com/itom-project/designerPlugins). For more information see the corresponding [section](https://itom-project.github.io/latest/docs/02_installation/build_dependencies.html) in the user documentation.
* [``itom``](https://itom-project.github.io/ "``itom``") is written in C++ and requires the Qt framework in version >5.6 (Qt6 is supported for version > 6.2). It is further dependent on OpenCV, the Point Cloud Library (optional) and Python 3.6 or higher including its important package [Numpy](https://numpy.org/doc/stable/index.html "Numpy").


## Contribution

You are welcome to use and test [``itom``](https://itom-project.github.io/ "``itom``"). If you want to you are invited to participate in the development of [``itom``](https://itom-project.github.io/ "``itom``") or some of its plugins. If you found any bug, feel free to post an [issue](https://github.com/itom-project/itom/issues "issue").

### pre-commit hooks
After the first cloning of the repositories, the [pre-commit](https://pre-commit.com/ "pre-commit") hooks should be installed once.
```bash
python -m pre_commit install
```
#### (optional) run against all files
It's usually a good idea to run the hooks against all of the files when adding new hooks (usually ``pre-commit`` will only run on the changed files during git hooks).
```bash
python -m pre_commit run --all-files
```

# Licensing
The core components and the main application of itom are covered by the [GNU Library General Public Licence (GNU LGPL)](https://github.com/itom-project/itom/blob/master/COPYING.txt "GNU Library General Public Licence (GNU LGPL)"). All components belonging to the SDK of [``itom``](https://itom-project.github.io/ "``itom``") (e.g. dataObject, pointCloud, addInInterface,…) are additionally covered by an [``itom``](https://itom-project.github.io/ "``itom``") exception. The main idea of this exception is to allow other libraries (e.g. plugins) to include and link agains components of itom SDK independent on the specific license model of the respective "other" library. All files belonging to the itom SDK are included in the folder SDK that is shipped with any setup or included in the build directory (when build from sources).

## itom Exception
The full text license of LGPL and itom [exception](https://github.com/itom-project/itom/blob/master/LGPL_EXCEPTION.txt "exception") is also included as file [COPYING](https://github.com/itom-project/itom/blob/master/COPYING.txt "COPYING") in the source distributions and setups.

All plugins and designer-plugins that can be integrated into itom can have their own licensing. Therefore the user is referred to the specific licensing documents or statements of each external library (plugin).

# Contact

[``itom``](https://itom-project.github.io/ "``itom``") is being developed since 2011 by

> [Institut für Technische Optik](http://www.uni-stuttgart.de/ito)

> University of Stuttgart

> Stuttgart

> Germany

in co-operation with 
> [twip Optical Solutions GmbH](http://www.twip-os.com)

> Stuttgart

> Germany
