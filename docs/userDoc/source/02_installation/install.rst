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

Alternatively, it is possible to get the sources of |itom| (e.g. clone the latest Git repository from https://bitbucket.org/itom/itom.git) and
compile an up-to-date version of |itom|. This is recommended for developers (e.g. plugin developers) and required of linux users. Before getting the source files,
check the build dependencies below which contain software packages and libraries necessary to build and |itom|.

.. toctree::
    :maxdepth: 1
    
    build_dependencies.rst
    git_clone.rst
    build_cmake.rst


Plugins, Designer-Plugins
**************************

Each plugin or designer plugin enhances the core-functionality of |itom| and is compiled in its own project. Therefore the installer or sources of |itom|
do no contain any plugin. Every plugin is distributed as a library file (*dll*, *so*,...) and - if necessary - other files.

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

   
   
   

   


