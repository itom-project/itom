.. include:: ../include/global.inc

Customize the menu and the Quickstart Bar
============================================

In this section the creation of buttons and menu entries which executes python code after clicking is described.
This creation is done by python code itseft.

You will find exemplary implementation within the toolbars delivered with itom-installer.

Add toolbar-buttons
---------------------------------------------------

Using the embedded scripting language in |itom|, you can add your own toolbars and buttons in order to automatically execute specific |python|-commands.
It is possible to create an arbitrary amount of different toolbars, where every toolbar is specified by an unique name. Then, you can add buttons to any desired
toolbar.

The following command will create one single button:

.. code-block:: python
    
    addButton(toolbarName, buttonName, python-code [, icon-filename])

The parameters are:

* toolbaName [string] is the name of the toolbar where this button should be placed. If the toolbar does not exist, it will be created and appended to the toolbars in |itom|.
* buttonName [string is the name of the button. This name is visible if no icon is assigned to this button (or if the icon can not be loaded or found).
* python-code [string] is the code-snippet executed if the button is pressed (currently no method or functions references are supported - unlike this is the case when creating menus).
* icon-filename [string, optional] is the path and filename to the icon (use png-files, ico-files are not supported), which should be used for this button. Please consider that relative pathes are searched in the application's current directory.

Please notice that any existing button with the same name that the new button and lying in the same toolbar will be deleted.

Examples for buttons:

.. code-block:: python
    :linenos:
    
    addButton("cameraTools", "start live image", "liveImage(cam)")
    addButton("cameraTools", "camera statistics", "print(cam.getParam('xsize'))\nprint(cam.getParam('ysize'))", "icons\\stats.png")
    absIcon = getAppPath() + 'Qitom/icon/attribute.png'
    addButton("otherTools", "new script", "newScript()")
    
Using itom delivered icons for buttons:

Itom comes with predefined buttons from ito, Qt and free icon data bases. You can address these icons like classic files via ':/pathname/filename'.
To select a suitable button use the iconBrowser (see :doc:`../script-language/debugging`)

.. code-block:: python
    :linenos:
    
    addButton("itoTools", "ITO", "print(\"Hello World\")", ":/application/icons/itomicon/itologo64.png")
    
    


Remove toolbar-buttons
--------------------------------

In order to remove any button added with the commands introduced in the section above, just use the command

.. code-block:: python
	
	removeButton(toolbarName, buttonName)

This command will delete the previously added button 'buttonName' in the toolbar with name 'toolbarName'. If this button was the last button in the specific toolbar, the
toolbar will be deleted, too.





	
