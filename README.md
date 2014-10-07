# itom #

welcome to the open source software **itom**. It allows operating measurement systems with multiple hardware components, like cameras, AD-converters, actuators, motor stages as well as handling your lab automation. The graphical user interface provides a quick and easy access to all components, complex measurement tasks and algorithms can be scripted using the embedded python scripting language and self-defined user interfaces finally provide a possibility to adapt **itom** to your special needs. External hardware or algorithms are added to **itom** by an integrated plugin system.

In order to learn more about **itom**, see the official homepage [itom.bitbucket.org](itom.bitbucket.org) or read the [user documentation](http://itom.bitbucket.org/latest/docs/)

### What is this repository for? ###

* This repository contains the free source code of the core application of **itom** that currently compiles on Windows and Linux.
* Many hardware and software plugins as well as special widget plugins (designer plugins) can be found in their respective repositories.

### How do I get set up? ###

* In order to get itom either download the ready-to-use setups for Windows 32bit and 64bit. Use the all-in-one installer in order to get itom including Python and some important Python packages or use the simple installer if you already have an appropriate version of Python installed on your computer.
* Clone this repositories and the [plugins](bitbucket.org/itom/plugins) and [designer plugins](bitbucket.org/itom/designerPlugins) repositories and compile by your self in order to get the latest version or develop own plugins. For more information see the corresponding [section](http://itom.bitbucket.org/latest/docs/02_installation/build_dependencies.html) in the user documentation.
* **itom** is written in C++ and requires the Qt framework in version 4.8 or higher (also 5.x). It is further dependent on OpenCV, the Point Cloud Library (optional), the source code editor QScintilla and Python 3.2 or higher including its important package Numpy.

### Contribution ###

You are welcome to use and test **itom**. If you want to you are invited to participate in the development of **itom** or some of its plugins. If you found any bug, feel free to post an issue.

### Contact ###

**itom** is being developped since 2011 by

> [Institut für Technische Optik](http://www.uni-stuttgart.de/ito)

> University of Stuttgart

> Stuttgart

> Germany

in co-operation with 
> [twip Optical Solutions GmbH](http://www.twip-os.com)

> Stuttgart

> Germany