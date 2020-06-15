.. include:: ../include/global.inc

.. _buildDocumentation:

Build documentation
==========================

Necessary tools
-----------------

In order to be able to build the documentation, you need to have some tools 
´installed on your computer:

1. Doxygen
    
    Doxygen is a source code documentation tool, that parses your C++-source 
    code and extracts the documentation strings for all methods.
    Additionally, it displays the entire class- and file-structure of your 
    project. |itom|'s user documentation sometimes uses results
    from Doxygen in order to show relevant C++-structures for programming plugins.
    
    Windows users can download the binaries as setup from http://www.doxygen.nl/download.html. 
    Under linux the easiest way is to get the latest
    doxygen package that is available for your distribution.

2. Python-Package Sphinx
    
    The real user documentation is created in |itom| using a python script that 
    needs the |Python| package **Sphinx** (http://sphinx-doc.org/).
    **Sphinx** itself requires other python packages to be installed on your 
    computer. For windows users, we therefore suggest to obtain **Sphinx** via
    the python package tools *pip* or *easy\_install*.
    
    The easiest way to obtain **Sphinx** is using the :ref:`Python package 
    manager <gui-pipmanager>` in the *script* menu of |itom|. Choose
    **install** and then **Sphinx** from the Python package index to obtain 
    Sphinx including all depending packages. Select the **Upgrade** checkbox
    if you want to upgrade **Sphinx** to a newer version.
    
    You can also manually download and install Sphinx and its depending packages. 
    Setup-versions of **Sphinx**, **Pygments**, **Jinja2**, **docutils**... are 
    also available from http://www.lfd.uci.edu/~gohlke/pythonlibs/. However, you 
    then need to separately install all depending packages of **Sphinx**.

3. Python-Packacke breathe
    
    Install the python package **breathe** (e.g. from pypi.org) in order to
    integrate some doxygen outputs into the Sphinx user documentation.

Run doxygen
---------------

In your build-directory of |itom|, you will find a folder **docs**. Open its 
subfolder **doxygen**. There you will find a document
**itom_doxygen.dox**. This document contains absolute paths to the source 
directories of |itom|'s sources. Run doxygen with this document
in order to create the source code documentation. 

On Windows computers, the easiest way to do this is open **itom_doxygen.dox** 
with the tool **doxywizard** that is located in the **bin**-folder 
of your **doxygen** installation. In **doxywizard** go to the *run*-tab and 
click on the *run*-button.

.. figure:: images/doxygen/doxywizard.png
    :alt: Doxywizard
    :scale: 100%
    :align: center

After the build process, a folder **xml** is created in the **doxygen** 
sub-folder of the **docs** folder. This **xml** folder is required afterwards.

Run Sphinx
--------------

Now, open |itom| and execute the script **create_doc.py** in the folder **docs/userDoc** 
of the build-directory. The default-builder of the
documentation is **qthelp**. If you also want to build the documentation for other 
builders, you can change the list *buildernames*. The following
values are possible::
    
    qthelp -> default qthelp format for opening the documentation within itom
    html -> creates the help for the homepage
    htmlhelp -> creates a chm-help format on Windows only
    latex -> creates a pdf-document using latex. You need to have latex 
      installed on your computer

The output of all build processes are located in the folder **docs/userDoc/build/<buildername>**. 
The locations of the Windows html-help generator or
the latex interpreter are detected when running **CMake** for the |itom|-project. 
The absolute paths to these tools are automatically
inserted into the script **create_doc.py**.

Show documentation in |itom|
-----------------------------

When clicking the *help*-button in |itom| or pressing **F1**, |Qt|'s assistant 
is opened with a set of documentation files. At first, |itom|
checks your |itom| installation for various documentation files. Their latest version 
is then copied into the **help** folder of the build-directory.
The search is executed for all **.qch**-files that are located in the **docs/userDoc**-directory.

After having copied the files, a collection-file is generated (containing all qch-files) 
and displayed in the assistant. If you have a setup version of |itom|, the help-folder 
already contains a compiled documentation file, that is displayed in this case. 
Please note that the file check is only execute once per |itom| session. 
Restart it in order to redo the check.