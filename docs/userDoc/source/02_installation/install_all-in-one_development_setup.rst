.. include:: ../include/global.inc

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