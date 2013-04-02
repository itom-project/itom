Basic ITOM-GUI
=================

Content:

.. toctree::
   :maxdepth: 1

   breakpoints
   filesystem
   globalvariables
   localvariables
   plugins
   editor

   
| After starting the software the GUI looks like in the following figure. 
| The GUI consists of the iTOM main frame and several so called Toolboxes_. In the middle of the main window is the console_, which can be used to type basic commands. The iTOM software and therefore also this console is based on python. So it's possible to use the standard python commands.
| In the following the several components of the GUI are explained.
|

.. figure:: images/iTOM.png

   Figure: iTOM Main Window
   
Toolboxes
==========

| It is possible to (un)dock the Toolboxes to the main frame at different positions. This is done by simple drag and drop of the titel bar of the toolboxes. Another way of (un)docking can be realized by double-clicking on the title bar.
| At the startup of the iTOM software all 5 Toolboxes are activated, which are:

- :doc:`breakpoints`
- :doc:`filesystem`
- :doc:`globalvariables`
- :doc:`localvariables`
- :doc:`plugins`

By right clicking on the main frame the toolboxes can be (de)activated.

.. figure:: images/toolboxmenu.png

   Figure: Toolboxmenu

Console
=======
Besides the toolboxes there is the main console, which is based on Python. The ">>" signs indicate that the console is ready for an input. Instead of typing all the commands in the console it's also possible to write python scripts  in the :doc:`editor`.

Main Buttons
=============
| At the startup of the software there are 4 buttons in the toolbar |mainsymbols|, whose functions are explained in the table below.
|

+-------------------+------------------+--------------------------------------------------------+
| Symbol            | Name (Shortcut)  | Description                                            |
+===================+==================+========================================================+
| |mainnew|         | New (CRTL+N)     | Opens a new Python script in the :doc:`editor`         |
+-------------------+------------------+--------------------------------------------------------+
| |mainopen|        | Open (CRTL+O)    | Opens a saved Python script in the :doc:`editor`       |
+-------------------+------------------+--------------------------------------------------------+
| |mainqtdesigner|  | Qt Designer      | Opens the Qt Designer, see :doc:`qtdesigner`           |
+-------------------+------------------+--------------------------------------------------------+
| |mainhelp|        | Help (F1)        | Opens this documentation                               |
+-------------------+------------------+--------------------------------------------------------+
| |debugOn|         | Run ... debug    | Toggle run GUI-embedded scripts in debug               |
+-------------------+------------------+--------------------------------------------------------+

.. |mainsymbols| image:: images/mainsymbols.png  
.. |mainnew| image:: images/mainnew.png  
.. |mainopen| image:: images/mainopen.png  
.. |mainqtdesigner| image:: images/mainqtdesigner.png  
.. |mainhelp| image:: images/mainhelp.png
.. |debugOn| image:: images/pythonDebug.png  

It is also possible to add Buttons to the iTOM GUI by the python commands. See 

Menu Structure
==============
In the menu line the following structure is given:

+----------------------------+-------------------------------------------------------------------+
|Structure                   |                                                                   |
+============+===============+===================================================================+
| **File**   | New Script    | Opens a new Python script in the :doc:`editor`                    |
+------------+---------------+-------------------------------------------------------------------+
|            | Open File     | Opens a saved Python script in the :doc:`editor`                  |
+------------+---------------+-------------------------------------------------------------------+
|            | Properties    | Basic Settings, see :ref:`set-itom-properties`                    |
+------------+---------------+-------------------------------------------------------------------+
| **Script** | stop          | Stop currently running python code                                |
+------------+---------------+-------------------------------------------------------------------+
|            | continue      | Run till end or next break point                                  |
+------------+---------------+-------------------------------------------------------------------+
|            | Step          | Step into call or if not possible make a stepover                 |
+------------+---------------+-------------------------------------------------------------------+
|            | Step over     | Step to next call in the current function                         |
+------------+---------------+-------------------------------------------------------------------+
|            | Step out      | Go on with code till next method call outside current function    |
+------------+---------------+-------------------------------------------------------------------+
|            | Run ... debug | Run GUI-embedded scripts (buttons) in debug (breakpoints enable)  |
+------------+---------------+-------------------------------------------------------------------+
|            | About Itom    | Reload imported python scripts (e.g. import numpy)                |
+------------+---------------+-------------------------------------------------------------------+
| **Help**   | Assistant     | Opens this documentation                                          |
+------------+---------------+-------------------------------------------------------------------+
|            | About Qt      | Information about Qt                                              |
+------------+---------------+-------------------------------------------------------------------+
|            | About Itom    | Information about your current ITOM-Version                       |
+------------+---------------+-------------------------------------------------------------------+











