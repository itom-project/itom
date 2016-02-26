'''create a dock widget form the main window at the bottom dock widget area
The dock widget can be destroyed by deleting the variable, that references to it.

If the dock widget is hidden by clicking the close button, it is hidden and can 
be re-shown e.g. by the context menu of the toolbar of itom'''
bottom_dock_window = ui("mainWindow.ui", ui.TYPEDOCKWIDGET, dockWidgetArea = ui.BOTTOMDOCKWIDGETAREA)

'''create a dock widget form the main window at the bottom dock widget area
The dock widget can be destroyed by deleting the variable, that references to it.

If the dock widget is hidden by clicking the close button, it is hidden and can 
be re-shown e.g. by the context menu of the toolbar of itom'''
right_dock_widget = ui("widget.ui", ui.TYPEDOCKWIDGET, dockWidgetArea = ui.RIGHTDOCKWIDGETAREA)

bottom_dock_window.show()
right_dock_widget.show()
