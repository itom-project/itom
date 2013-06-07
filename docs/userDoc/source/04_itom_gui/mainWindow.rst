.. include:: ../include/global.inc

Main Window
============

When starting |itom| a splash screen is shown during the loading process of many components and external plugins. Then, the main window appears like in the following figure:

.. figure:: images/iTOM.png
    :scale: 100%
    :align: center

The GUI consists of a command line widget (console_) in the center and several toolboxes_. |itom| contains the embedded scripting language Python 3. Simple python commands can therefore be typed in the console. The
toolboxes provide many further functionalities of |itom|. Additionally, every opened actuator or dataIO plugin may provide its own toolbox, that can also be docked into |itom|'s main window.

Besides the console and toolboxes, further important functionalities are provided by the menu or the main toolbars.

Console
---------

The command line in the center of the main window allows to execute single or multi-line python commands. Additionally all messages, warnings and errors coming from python methods or |itom| itself are printed in that console
widget, where error messages are highlighted with a red background.

.. figure:: images/consoleError.png
    :scale: 100%
    :align: center

Usually, the last line of the command line shows the ">>" sign, that indicates that the console is ready for a new input. You can either write a single python command and press the return key in order to execute it or you can write multiline commands. In order to create a new line for this, press the Shift+Return keys, which is a smooth line break (known from other software). After the final command simply press the return key such that the whole
command block is executed.

.. figure:: images/consoleMultiLine.png
    :scale: 100%
    :align: center

The current line or code-block that is executed is highlighted with a yellow background. For multi-line commands, |itom| parses the whole command block and divides it into logical blocks, such that the highlighed background
switches from one block to the other one.

In the command line you can use every python command however the command line is not able to handle inputs. Additionally you can use one of the following key-words in order to clear the command line::
    
    clc
    clear

Instead of typing all commands in the console, write your entire python scripts in the :doc:`editor`.

Main menus and toolbars
-------------------------

This is an overview about the menu structure of itom

+------------+--------------------+----------------------+------------------------------------------------------------------------------+
|Structure   |                    |                      |                                                                              |
+============+====================+======================+==============================================================================+
| **File**   | New Script         | |mainnew| Ctrl+N     | Opens a new Python script in the :doc:`editor`                               |
+------------+--------------------+----------------------+------------------------------------------------------------------------------+
|            | Open File          | |mainopen| Ctrl+O    | Opens a saved Python script in the :doc:`editor`                             |
+------------+--------------------+----------------------+------------------------------------------------------------------------------+
|            | Properties...      |                      | property dialog for important settings of itom                               |
+------------+--------------------+----------------------+------------------------------------------------------------------------------+
|            | User Management... |                      | organizing the user dependent appearance of itom                             |
+------------+--------------------+----------------------+------------------------------------------------------------------------------+
|            | Loaded plugins...  |                      | opens dialog to see the load status of all plugins                           |
+------------+--------------------+----------------------+------------------------------------------------------------------------------+
|            | Exit               |                      | terminates and quits itom                                                    |
+------------+--------------------+----------------------+------------------------------------------------------------------------------+
| **View**   | Toolboxes          |                      | In the submenu you can toggle the visibility of all toolbars and -boxes      |
+------------+--------------------+----------------------+------------------------------------------------------------------------------+
| **Script** | stop               | Shift+F10            | Stop currently running python code                                           |
+------------+--------------------+----------------------+------------------------------------------------------------------------------+
|            | continue           | F6                   | Run till end or next break point                                             |
+------------+--------------------+----------------------+------------------------------------------------------------------------------+
|            | Step               | F11                  | Step into call or if not possible make a stepover                            |
+------------+--------------------+----------------------+------------------------------------------------------------------------------+
|            | Step over          | F10                  | Step to next call in the current function                                    |
+------------+--------------------+----------------------+------------------------------------------------------------------------------+
|            | Step out           | Shift+F11            | Go on with code till next method call outside current function               |
+------------+--------------------+----------------------+------------------------------------------------------------------------------+
|            | Run ... debug      | |debugOn|            | If toggled python methods triggered by user interfaces is executed in debug. |
+------------+--------------------+----------------------+------------------------------------------------------------------------------+
|            | Reload modules     |                      | Reload imported python packages (e.g. import numpy)                          |
+------------+--------------------+----------------------+------------------------------------------------------------------------------+
| **Help**   | Assistant          | |mainhelp| F1        | Opens this documentation                                                     |
+------------+--------------------+----------------------+------------------------------------------------------------------------------+
|            | About Qt           |                      | Information about Qt                                                         |
+------------+--------------------+----------------------+------------------------------------------------------------------------------+
|            | About Itom         |                      | Information about your current itom-Version                                  |
+------------+--------------------+----------------------+------------------------------------------------------------------------------+
| **Tools**  | UI Designer        | |mainqtdesigner|     | Opens the Qt Designer (see :ref:`qtdesigner`)                                |
+------------+--------------------+----------------------+------------------------------------------------------------------------------+

.. |mainsymbols| image:: images/mainsymbols.png  
.. |mainnew| image:: images/mainnew.png  
.. |mainopen| image:: images/mainopen.png  
.. |mainqtdesigner| image:: images/mainqtdesigner.png  
.. |mainhelp| image:: images/mainhelp.png
.. |debugOn| image:: images/pythonDebug.png 

All the actions mentioned in the table above are accessible by either the menu of |itom| or some of them are also available in some main toolbars that come with |itom|.
   
Toolboxes
-----------

Content:

.. toctree::
   :maxdepth: 1

   workspace
   plugins
   filesystem
   breakpoints

The core application of |itom| already comes with a set of different toolboxes. Additionally, many plugins provide the possibility to open a toolbox for every opened hardware instance, like actuators or cameras.
These toolboxes are then also inserted into the main window of |itom|. Usually toolboxes can be docked into the main window or be in a floating state, such they appear like an unresizable window. If docked, they
can be positioned at the left or right side or the top or bottom of the main window. Some of them however are limited with respect to the dockable positions.

All available toolboxes are listed in the menu **View >> Toolboxes**, where hidden toolboxes can be shown again. Additionally, a right click to any place in the toolbar openes the following context menu where
the first items also access the loaded toolboxes. The items after the separator correspond to the toolbars, such that they can be hidden or shown:

.. figure:: images/toolboxmenu.png
    :scale: 100%
    :align: center

| It is possible to (un)dock the Toolboxes to the main frame at different positions. This is done by simple drag and drop of the titel bar of the toolboxes. Another way of (un)docking can be realized by double-clicking on the title bar.
| At the startup of the iTOM software all 5 Toolboxes are activated, which are:

The following main toolbars are available:

- :doc:`workspace` shows the global and local workspaces of Python.
- :doc:`plugins` shows all loaded plugins including opened instances.
- :doc:`filesystem` gives you access to the file system of your harddrive.
- :doc:`breakpoints` shows all breakpoints added to Python scripts.
- **callstack** shows the callstack when the Python script execution stops at a breakpoint.










