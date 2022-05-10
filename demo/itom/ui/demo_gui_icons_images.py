"""GUI icons as images
======================

In this example, all icons are loaded in the UI file
as relative path to the icons in this or the icons subfolder.
"""
from itom import ui

# sphinx_gallery_thumbnail_path = '11_demos/_static/_thumb/demoGUIImages.png'


def closeGui():
    gui.hide()


gui = ui("gui_icons_images.ui", type=ui.TYPEWINDOW)
gui.actionClose.connect("triggered()", closeGui)
gui.show()

###############################################################################
# .. image:: ../../_static/demoGUIIconsImages_1.png
#    :width: 50%
