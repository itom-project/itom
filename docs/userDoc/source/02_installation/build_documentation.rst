.. include:: ../include/global.inc

Build documentation
==========================

Necessary tools
-----------------

In order to be able to build the documentation, you need to have some tools installed on your computer:

1. Doxygen
    
    Doxygen is a source code documentation tool, that parses your C++-source code and extracts the documentation strings for all methods.
    Additionally, it displays the entire class- and file-structure of your project. |itom|'s user documentation sometimes uses results
    from Doxygen in order to show relevant C++-structures for programming plugins.
    
    You will find **Doxygen** under http://www.stack.nl/~dimitri/doxygen/. Goto the download section and get the latest release. Windows
    users can directly download the binaries provided as setup version. Install **Doxygen** on your computer.

2. Python-Package Sphinx
    
    The real user documentation is created in |itom| using a python script that needs the |Python| package **Sphinx** (http://sphinx-doc.org/).
    **Sphinx** itself requires other python packages installed on your computer. Therefore we suggest to get **Sphinx** by the python package
    **distribute** that is able to download the latest version of **Sphinx** including all depending packages. For the Windows operating systems,
    **distribute** also installs an application **easy_install** in the **Python\Scripts** directory. **easy_install** can be called from any
    command line in order to get **Sphinx**.
    
    In order to get **distribute** and **easy_install**, follow one of the following two possibilities:
    
    1. Easy installation by distribute_setup.py:
        
        * Go to http://python-distribute.org and download the script **distribute_setup.py**. 
        * Execute the script with your Python 3 application, e.g. by typing::
            
            C:/python32/python.exe distribute_setup.py
        
        in your command line where you previously moved into the directory where you saved the file. The package distribute is then automatically downloaded from the Python package manager. Replace **C:/python32** by the path to your Python installation.
        
    2. Official way:
        
        * Go to http://pypi.python.org/pypi/distribute and download the file *distribute-x.x.xx.tar.gz* and unzip to any folder on your computer.
        * Open a command line (cmd) and change to the folder (command **cd**) where you unpacked **distribute**.
        * Execute the following command (Replace with your correct python-directory)::
            
            C:/python32/python.exe distribute_setup.py
            
        * Now distribute is installed. When done, you can delete the folder the files were extracted to.
        * Verify that the application **easy_install.exe** is now available in your *python/scripts* directory (Windows only)
    
    Now you can use **easy_install** in order to get **Sphinx**:
    
    * Open a command line (cmd) and change to the subfolder **Scripts** of your |Python| installation.
    * Execute the following command::
        
        easy_install -U Sphinx
        
    * Now Sphinx is downloaded and Sphinx and its depending packages are installed.
    
    You can also manually download and install Sphinx and its depending packages. Setup-versions of **Sphinx**, **Pygments**, **Jinja2**, **docutils**... are also available from http://www.lfd.uci.edu/~gohlke/pythonlibs/. However, you then need to separately install all
    depending packages of **Sphinx**.

Run doxygen
---------------

In your build-directory of |itom|, you will find a folder **docs**. Open its subfolder **doxygen**. There you will find a document
**itom_doxygen.dox**. This document contains absolute pathes to the source directories of |itom|'s sources. Run doxygen with this document
in order to create the source code documentation. 

On Windows computers, the easiest way to do this is open **itom_doxygen.dox** with the tool
**doxywizard** that lies in the **bin**-folder of you **doxygen** installation. In **doxywizard** go to the *run*-tab and click on the *run*-button.

.. figure:: images/doxygen/doxywizard.png
    :alt: Doxywizard
    :scale: 70%

After the build process, a folder **xml** is created in the
**doxygen** subfolder of the **docs** folder. This **xml** folder is required afterwards.

Run Sphinx
--------------

Now open |itom| and execute the script **create_doc.py** in the folder **docs/userDoc** of the build-directory. The default-builder of the
documentation is **qthelp**. If you also want to build the documentation for other builders, you can change the list *buildernames*. The following
values are possible::
    
    qthelp -> default qthelp format for opening the documentation within itom
    htmlhelp -> creates a chm-help format on Windows only
    latex -> creates a pdf-document using latex. You need to have latex installed on your computer

The output of all build processes are located in the folder **docs/userDoc/build/<buildername>**. The locations of the Windows html-help generator or
the latex interpreter are detected when running **CMake** for the |itom|-project. The absolute paths to these tools are automatically
inserted into the script **create_doc.py**.

Show documentation in |itom|
-----------------------------

When clicking the *help*-button in |itom| or pressing **F1**, |Qt|'s assistant is opened with a set of documentation files. At first, |itom|
checks your |itom| installation for various documentation files. Their latest version is the copied into the **help** folder of the build-directory.
The search is executed for all **.qch**-files that are located in the **docs/userDoc**-directory and in the folder **plugins** or any subfolder.

After having copied the files, a collection-file is generated (containing all qch-files) and displayed in the assistant. If you have a setup version of |itom|, the help-folder already contains a compiled documentation file, that is displayed in this case.