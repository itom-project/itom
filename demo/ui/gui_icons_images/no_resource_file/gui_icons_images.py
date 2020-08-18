# coding=iso-8859-15

"""In this example, all icons are loaded in the UI file
as relative path to the icons in this or the icons subfolder.
"""

def closeGui():
    gui.hide()

gui = ui("gui_icons_images.ui", type=ui.TYPEWINDOW)
gui.actionClose.connect("triggered()", closeGui)
gui.show()