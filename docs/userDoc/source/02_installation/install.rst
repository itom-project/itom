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
    * Linux based OS (tested with Debian and Ubuntu)
    * Mac OS X 10.9 (Mavericks) or later

* Processor architectur
    
    * Windows and Linux: 32 or 64bit processor architecture, SSE and SSE2 support
    * Max OS X: All Intel-based Macs with an Intel Core 2 or later

* 800MHz processor (dual-core recommended)
* 512MB of RAM
* 1024 x 768 screen resolution
* 200+ MB hard drive space

Installation from setup
************************

Currently, there are both a *32bit* and a *64bit* setup version of |itom| available (Windows only). For linux users there is no pre-compiled package available.
The use of the setup version is recommended for users that mainly want to work with |itom| without developing new algorithms or plugins. The setup
can be downloaded from https://bitbucket.org/itom/itom/downloads. The setup intalls the core application, a number of useful hardware plugins and the designer plugins which provide plotting functionalities to |itom|.

.. toctree::
    :maxdepth: 1
    
    install_windows_setup.rst


Build from Sources
************************ 

.. note::
    
    There is a all-in-one build development installation available (Windows, Visual Studio 2010 only). This contains all dependencies that are required to build itom and its main plugins
    from sources. See the section `All-In-One development setup` below for more information.

Alternatively, it is possible to get the sources of |itom| (e.g. clone the latest Git repository from https://bitbucket.org/itom/itom.git) and
compile an up-to-date version of |itom|. This is recommended for developers (e.g. plugin developers) and required for linux users. Before getting the source files,
check the build dependencies below which contain software packages and libraries necessary to build and |itom|.

.. toctree::
    :maxdepth: 1
    
    build_dependencies.rst
    git_clone.rst
    build_cmake.rst

For linux as well as Mac OS X, a short description of the installation that contains more specific information than the sections above, is available here:

.. toctree::
    :maxdepth: 1
    
    build_linux.rst
    build_osx.rst


Plugins, Designer-Plugins
**************************

Each plugin or designer plugin enhances the core-functionality of |itom| and is compiled in its own project. Therefore the installer or sources of |itom|
do no contain any plugin. Every plugin is distributed as a library file (*dll*, *so*,...) and - if necessary - other files.

All-In-One development setup
******************************

For users who want to get a development environment for itom, the main plugins and designer plugins there is an all-in-one development setup available.

Using this setup, you only need to unzip one or two archives to your harddrive, install **git** and **Python** that are included in this archive and
execute a setup script, written in Python. This script automatically downloads the current sources of |itom| and its plugins from the internet,
configures the 3rd party dependencies (also provided in this package) and automatically configures and generates CMake for the single repositories.
Using this setup tool, you can start developing |itom| or plugins within a short time.

For more information see:

.. toctree::
    :maxdepth: 1
    
    all-in-one_development_setup.rst

Get this help
***********************

The user documentation of |itom| can be distributed in various formats. The main format is called **qthelp**, such that the documentation is
displayed in the |Qt|-Assistant that is directly accessible from the |itom| GUI. On windows PCs it is possible to compile the help as a Windows Help
Document (chm) or to create latex-files from the help, in order to create a pdf-document. The base format of all these
formats is a collection of *html*-pages.

If you compiled |itom| from sources, no compiled documentation file is provided. Therefore, you need to compile the help by yourself:

.. toctree::
    :maxdepth: 1
    
    build_documentation.rst
    plugin_documentation.rst

   
   
   

   


