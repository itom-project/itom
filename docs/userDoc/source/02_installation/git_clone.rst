.. include:: ../include/global.inc

.. _install-git-clone:

Get sources from Git
==========================

GIT is a distributed version and source code management system, that allows several developers to work on the same source code simultaneously. |itom| is hosted under GIT
at bitbucket.org.

For ease of development the central |itomproject| repository was established, which comprises the the core of |itom|,
some plugins as well as designer-plugins as submodules, which are otherwise hosted in different Git-repositories on *bitbucket.org*.

To clone from the central |itomproject| and initialize the submodules correctly, please follow these steps:

.. code-block:: bash

    git clone git@bitbucket.org:itom/itomproject.git
    cd itomproject
    git submodule init
    git submodule update

This will automatically clone the related submodules |itom| (core), |plugins| and |designerplugins|.

If needed the different repsoitories can be found at the following resources:

================== ================================================ =================================================== ================================================
Project            Website                                          Git-Repository (https)                              Git-Repository (ssh)
================== ================================================ =================================================== ================================================
|itomproject|      https://bitbucket.org/itom/itomproject           https://bitbucket.org/itom/itomproject.git          git@bitbucket.org:itom/itomproject.git
|itom| (core)      https://bitbucket.org/itom/itom                  https://bitbucket.org/itom/itom.git                 git@bitbucket.org:itom/itom.git
|plugins|          https://bitbucket.org/itom/plugins               https://bitbucket.org/itom/plugins.git              git@bitbucket.org:itom/plugins.git
|designerplugins|  https://bitbucket.org/itom/designerplugins       https://bitbucket.org/itom/designerplugins.git      git@bitbucket.org:itom/designerplugins.git
================== ================================================ =================================================== ================================================

In order to get the sources you want, clone the specific repository to a source-folder on your computer. If you want to receive data by *ssh*, you need your own
account on *bitbucket.org*, where you public *ssh* has been set under your personal settings.

.. code-block:: bash

    git clone https://bitbucket.org/itom/itom


If you are using the the central |itomproject| repository to build the source code, please mind that the submodule are listed as *ssh* repositories.
To make use of the *https* protocol you need to replace the submodule source designation in the .gitmodule file.
