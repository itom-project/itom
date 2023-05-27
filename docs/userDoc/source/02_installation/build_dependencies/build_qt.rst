:orphan:

.. include:: ../../include/global.inc
.. |star| unicode:: U+002A

.. _build_dependencies_qt:


Install Qt
==========

Qt 5
----

Creating prebuild version of Qt:
For a working |itom| development environment only a prebuild version of Qt is necessary.

* Install Qt into the **${MAINDIR}**/3rdParty/Qt5.12.1 with the components
    msvc2017 64-bit, Qt WebView, Qt WebEngine. Qt Creator is not necessary,
    but can not be unchecked
* After the installation copy the folder to another location
    (**${MAINDIR}**/3rdParty/Qt5.12.1_backup) and uninstall Qt
    in the Windows program settings
* Rename Qt5.12.1_backup back to Qt5.12.1
* From Qt5.12.1/Docs/Qt-5.12 copy all |star| .qch files (only in the main folder)
    to Qt5.12.1/5.12/msvc2017_64/doc
* Start the Qt Assistant (**${MAINDIR}**/3rdParty/Qt5.12.1/5.12/msvc2017_64/bin),
    open **options/documentation** and delete all. Add then the copied documentation files.
* From Qt5.12.1 delete the following things:
    * folder: dist, Examples, Tools, vcredist, Docs (after having copied the qch files)
    * files: all files in the main folder, e. g. components.xml...
* Copy OpenSSL **libeay32.dll** and **ssleay32.dll** to the **${MAINDIR}**/3rdParty/Qt5.12.1/5.12/msvc2017_64/bin

.. warning::

    Create a path on your hard drive with a long, long path name (called ${MAINDIR})
    (later, the all-in-one path on destination computers must be shorter than
    this path name, due to the Qt patching)

.. warning::

    The QT version **5.6.2** has a bug which prevent the start of the QT designer
    {'QTBUG-53984': ('https://bugreports.qt.io/browse/QTBUG-53984', 'QTBUG-53984')}.
    The workaround is to change the name of **Qt5WebEngineWidgets.dll** and
    **Qt5WebEngineWidgetsd.dll**, then copy the **Qt5Core.dll** and **Qt5Cored.dll**
    and change the name of these dll-files into **Qt5WebEngineWidgets.dll** and
    **Qt5WebEngineWidgetsd.dll**. This bug should be solved with QT version 5.6.3
    (release August 2017).

.. warning::

    **Qt WebEngine**, **Qt WebEngineWidgets** are only available under VS 2017 as it is shown in the figure below!

    Qt 5.10.1 supports **Qt WebEngine**, **Qt WebEngineWidgets** for VS2015.

.. figure:: ../images/all-in-one-create/QT_WebEngine_hint.png
    :scale: 100%
    :align: center

Qt 6
----

Copy the subfolders of the doc folder into the folder /X.X.X//msvc2017_64/doc. Change in the preferences of the Assistant the documentation folder.
