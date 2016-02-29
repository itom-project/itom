.. include:: ../include/global.inc

.. _gui-propertydialog:

Property Dialog
******************

The property dialog stores the main itom settings, including all the widgets settings. The Property dialog can be found by clicking "File -> "Properties...". It has different sections with subsections corresponding to the sections on this help page. 

General
===============

Application
-------------

* If the "show message before closing the application" checkbox is checked, the application will ask you if you really want to close Itom.

Help Viewer
-------------

This property section is responsible for the behaviour of the "Help" dialog. If the help widget is hidden in you mainwindow, go to View -> Toolboxes -> Help in the main toolbar.

.. figure:: images/propGeneralHelpViewer.png
    :scale: 100%
    :align: center

Local and remote databases
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Most help files are organized in databases. To display these files, the green underlined checkbox has to be checked. To manage, update and load new databases the green box offers a variety of options. Each database listed underneath "Local" are saved on the harddrive. The last column shows if there are any online updates available. To refresh the updatestate of the databases, just click the "refresh" button above. 

If the the internet connection is very slow a timeout error might appear during updates. In this case increase the timeout time and check you internet connection.

Generated help files
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The Algorithms, Widgets, DataIO and Actuator help files are dynamically created during runtime. These help files are displayed when the corresponding checkboxes in the red box are checked.

Language
-------------

By selecting a language inside the list box and clicking the "Apply" button a new language for itom is selected. Itom must be restarted to load the new language.

Console
===============

Command History
--------------------------

These options refer to the command history widget that is available under View -> Toolboxes -> Command History. 

Line Wrap
--------------------------

The first three radio buttons manage when a line is wrapped. The group box underneath offers the possibility to display small icons at a line wrap and to indent the next line. The lowest group box offers three modes how to indent the wrapped line.  

Python
===============

Startup 
-------------

By "Add File" is is possible to add python files that are executed when itom is started. It is kind of like the autostart folder in Windows.

Editor
===============

.. _gui-prop-py-api:

API
-------------

The api files listed in the checkbox are necessary for syntax highlighting. New api files can be added by clicking on the "Add API" button on the right side.

.. _gui-prop-auto-completion:

Auto Completion
--------------------------

.. figure:: images/propEditorAutoCompletion.png
    :scale: 100%
    :align: center

The auto completion has two main functions. It offers available commands after entering some characters (number of minimum characters can be set in the "threshold" spin box).

.. figure:: images/propEditorAutoCompletion_2.png
    :scale: 100%
    :align: center

The other function  shows a list of available members of classes after entering a dot. 

.. figure:: images/propEditorAutoCompletion_3.png
    :scale: 100%
    :align: center

The three radio buttons in the group box at the bottom of the page set the source of the auto completion. Therefore take a look at API.

.. _gui-prop-calltips:

Calltips
-------------

Calltips are tooltips that appear to display arguments of functions. They appear after entering "(". The number of calltips is important if there are overloaded functions with different parameter sets.

.. figure:: images/propEditorCalltips.png
    :scale: 100%
    :align: center

.. _gui-prop-py-general:
    
General
-------------

The first group box manages the indentation. 

* "Auto indentation" automatically indents a new block after an "if ():" or a for loop occurred. 
* if "Use tabs for indentation" is checked, tabs are used, otherwise spaces.
* 
* "Show Whitespaces" displays small light grey dots in each indentation.
* The "Indentation Width" spinbox sets the standard width for the indentation

Inside "Indentation Warning" group box it is possible to select which kind of indentation is marked as wrong. Make sure not to create a conflict with the checkboxes listed above ("use Tabs for Indentation"). The following image shows a warning caused by wrong indentation (tabs).

.. figure:: images/propEditorGeneral.png
    :scale: 100%
    :align: center

The radio buttons inside the "End-of-line (EOL) mode" group box decide whether to use "", "" or "" as eol, depending on your operating system.

The **Python Syntax Checker** checks the code inside the editor widget for bugs. If there are bug, a small red ladybug is shown besides the line numbers. If the cursor is moved over a ladybug, a tooltip shows the error (for more information see the help about the :ref:`script editor window <gui-editor-syntax-check>`).

* The Itom module is always included in every script. This causes wrong bugs appearances because the checking module (frosted) is not able to see the itom inclusion. To avoid these errors check the "Automatically include itom module..." check box. It includes "include itom" in every header before checking the code to avoid wrong bugs. 

* The "Check interval" check box sets the interval the code is send to "frosted" for syntax checks. 

The **class navigator** feature allows configuring the :ref:`class navigator <gui-editor-class-navigator>` of any script editor window. The checkbox of the entire groupbox en- or disables this feature. Use the timer to recheck the script structure after a certain amount of seconds since the last change of the script. If the timer is disabled, the structure is only analyzed when the script is shown or loaded.


.. _gui-prop-py-styles:

Styles
-------------

This page is responsible for the highlighting of reserved words, comments, identifier and so on. The style for each type of text, listed in the listbox, can be set individually. 

Plots and Figures
==============================

.. _gui-default-plots:

Default Plots
--------------------------

* The first table lists all available plugins to plot data. The different columns show what kind of input data they accept and what they should be used for.

* The second table shows different categories for plots. For each category a default plugin can be selected. This default plugin will be used to plot the incoming data. To change the standard plugin, double click the last column. 

.. figure:: images/propPlotsDefaultPlots.png
    :scale: 100%
    :align: center






