.. include:: ../include/global.inc

Installation
################

In this section we want to show you how to get and install |itom| on your computer.

Minimum system requirements
****************************
Before installation, please review the minimum system requirements.

* Operating System

    * Windows XP SP3 *or*
    * Windows Vista SP1 (SP2 and platform update recommended) *or*
    * Windows 7 *or*
    * Windows 8 (not tested yet) *or*
    * Linux based OS (tested with Debian)

* 32 or 64bit processor architecture
* 800MHz processor (dual-core recommended)
* 512MB of RAM
* 1024 x 768 screen resolution
* 200+ MB hard drive space



Getting |itom|
***************

Depending on your operating system, there are currently two different possibilites how to obtain and run |itom| on your computer.

Setup
-----

At first, windows users that mainly want to use |itom| and pre-compiled plugins can download the latest *32bit* or *64bit* setup of |itom|
and install the current release on the computer (see :doc:`setup.exe <install_windows_setup>`). For linux users there is currently no
pre-compiled package available. The setup can for instance be downloaded from https://bitbucket.org/itom/itom/downloads. Please consider that
the core installation of |itom| does not contain any hardware or software plugins and no designer plugins which provide plotting functionalities to |itom|.

.. toctree::
    :maxdepth: 1
    
    install_windows_setup.rst

Build from Sources
------------------    

Secondly, it is possible to get the sources of |itom| (e.g. clone the latest Git repository from https://bitbucket.org/itom/itom.git) and
compile an up-to-date version of |itom|. This is recommended for developers (e.g. plugin developers) and linux users. Before getting the source files,
check the build dependencies below which contain software packages and libraries necessary to build and |itom|.

.. toctree::
    :maxdepth: 1
    
    build_dependencies.rst


Plugins, Designer-Plugins
**************************

Each plugin or designer plugin enhances the core-functionality of |itom| and is compiled in its own project. Therefore the installer or sources of |itom|
do no contain any plugin. Every plugin is distributed as a library file (*dll*, *so*,...) and - if necessary - other files.

   
   
   

   


