# coding=iso-8859-15

"""Demostration to show a ui file, that contains icons from a qrc / rcc
resource file.

To create the ui file, you need at first the icons and the index
file qrc, that lists all the icons and gives them a unique path.

This qrc file then must be compiled together with the contained
icons to a rcc resource file using the rcc binary (shipped with itom).

For instance, to compile the qrc file in the icons subfolder to
a rcc file in this folder, navigate a command line to the icons
subfolder and apply

rcc -binary myIconResource.qrc -o ../myIconResource.rcc

Then, the resource file must be loaded first into itom before opening
the ui.
"""

import itom

# load the resource file, such that the contained icons can be used in future UIs
itom.registerResource("myIconResource.rcc")

# load the UI (open the UI in QtDesigner to see how the icons are assigned there)
gui = ui("gui_with_icons_from_resource.ui", type=ui.TYPEWINDOW)

# show the UI (no functionality at all)
gui.show()