.. include:: ../include/global.inc

Get this help
***********************

The user documentation of |itom| can be distributed in various formats. The main
format is called **qthelp**, such that the documentation is
displayed in the |Qt|-Assistant that is directly accessible from the |itom| GUI.
On windows PCs it is possible to compile the help as a Windows Help
Document (chm) or to create latex-files from the help, in order to create a
pdf-document. The base format of all these formats is a collection of *html*-pages.

If you compiled |itom| from sources, no compiled documentation file is provided.
Therefore, you need to compile the help by yourself:

.. toctree::
    :maxdepth: 1

    build_documentation.rst
    plugin_documentation.rst
