.. include:: ../include/global.inc

Build from Sources
************************

.. note::

    There is a all-in-one build development installation available (Windows,
    several versions of Visual Studio). This contains all dependencies that are
    required to build itom and its main plugins
    from sources. See the section :ref:`All-In-One development setup
    <install-all-in-one-dev-setup>` below for more information.

Alternatively, it is possible to get the sources of |itomproject| (e.g. clone the latest
Git repository from git@github.com:itom-project/itomProject.git), initialize and update the
submodules **itom**, **plugins** and **designerplugins**.

.. code-block:: bash
    git clone --recursive --remote git@github.com:itom-project/itomProject.git
    cd itomproject
    git submodule foreach --recursive git checkout master

Then you can compile an up-to-date version of |itomproject|. This is recommended for developers
(e.g. plugin developers) and required for linux users. Before getting the source files,
check the build dependencies below which contain software packages and libraries
necessary to build and |itomproject|.

.. toctree::
    :maxdepth: 2

    build_dependencies.rst
    git_clone.rst
    build_cmake.rst
    build_cmake_vars.rst

For linux including the Raspberry Pi as well as Mac OS X, a short description of
the installation that contains more specific information than the sections above,
is available here:

.. toctree::
    :maxdepth: 2

    build_debian.rst
    build_fedora.rst
    build_centos.rst
    build_osx.rst
