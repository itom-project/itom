"""GUI icons from resources
===========================

Demonstration to show a ui file, that contains icons from a qrc / rcc
resource file.

To create the ui file, you need at first the icons and the index
file qrc, that lists all the icons and gives them a unique path.

This qrc file then must be compiled together with the contained
icons to a rcc resource file using the rcc binary (shipped with itom).

For instance, to compile the qrc file in the icons subfolder to
a rcc file in this folder, navigate a command line to the icons
subfolder and apply

..code-block:: bat

    rcc -binary myIconResource.qrc -o ../myIconResource.rcc

Then, the resource file must be loaded first into itom before opening
the ui.
"""

import itom
from itom import ui

# sphinx_gallery_thumbnail_path = '11_demos/_static/_thumb/demoGUIResource.png'


# load the resource file, such that the contained icons can be used in future UIs
itom.registerResource("myIconResource.rcc")

# load the UI (open the UI in QtDesigner to see how the icons are assigned there)
gui = ui("gui_with_icons_from_resource.ui", type=ui.TYPEWINDOW)

# show the UI (no functionality at all)
gui.show()

###############################################################################
# .. image:: ../../_static/demoGUIIconsResources_1.png
#    :width: 50%
