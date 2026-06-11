.. include:: ../include/global.inc

.. _install-git-clone:

Get sources from Git
==========================

GIT is a distributed version and source code management system, that allows several developers to work on the same source code simultaneously. |itom| is hosted under GIT
at `GitHub <https://github.com>`_.

For ease of development the central **itomproject** repository was established, which comprises the the core of |itom|,
some plugins as well as designer-plugins as submodules, which are otherwise hosted in different Git-repositories on `GitHub <https://github.com>`_.

To clone from the central **itomproject** and initialize the submodules correctly, please follow these steps:

.. code-block:: bash

    git clone --recursive --remote git@github.com:itom-project/itomProject.git
    cd itomproject
    git submodule foreach --recursive git checkout master

This will automatically clone the related submodules |itom| (core), **plugins** and **designerplugins**.

If needed the different repsoitories can be found at the following resources:

+---------------------+---------------------------------------------------+-----------------------------------------------------+------------------------------------------------------+
| Project             | Website                                           | Git Repository (HTTPS)                              | Git Repository (SSH)                                 |
+=====================+===================================================+=====================================================+======================================================+
| **itomproject**     | https://github.com/itom-project/itomProject       | https://github.com/itom-project/itomProject.git     | `<git@github.com:itom-project/itomProject.git>`_     |
+---------------------+---------------------------------------------------+-----------------------------------------------------+------------------------------------------------------+
| |itom| (core)       | https://github.com/itom-project/itom              | https://github.com/itom-project/itom.git            | `<git@github.com:itom-project/itom.git>`_            |
+---------------------+---------------------------------------------------+-----------------------------------------------------+------------------------------------------------------+
| **plugins**         | https://github.com/itom-project/plugins           | https://github.com/itom-project/plugins.git         | `<git@github.com:itom-project/plugins.git>`_         |
+---------------------+---------------------------------------------------+-----------------------------------------------------+------------------------------------------------------+
| **designerplugins** | https://github.com/itom-project/designerPlugins   | https://github.com/itom-project/designerPlugins.git | `<git@github.com:itom-project/designerPlugins.git>`_ |
+---------------------+---------------------------------------------------+-----------------------------------------------------+------------------------------------------------------+

In order to get the sources you want, clone the specific repository to a source-folder on your computer. If you want to receive data by *ssh*, you need your own
account on `GitHub <https://github.com>`_, where you public *ssh* has been set under your personal settings.

.. code-block:: bash

    git clone git@github.com:itom-project/itom.git


If you are using the the central **itomproject** repository to build the source code, please mind that the submodule are listed as *ssh* repositories.
To make use of the *https* protocol you need to replace the submodule source designation in the .gitmodule file.
