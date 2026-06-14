"""Splitter
===========

This demo shows how to manipulate a splitter, set as layout in QtDesigner.

.. hint::

    This demo uses specially wrapped methods of QSplitter. For more information see
    section ``Calling slots`` in https://itom-project.github.io/latest/docs/06_extended_gui/qtdesigner.html)
    These methods are indiciated by #-> special method call
"""
from itomUi import (
    ItomUi,
)  # import the base class ItomUi from the module itomUi in the itom-packages subfolder
from itom import ui

# sphinx_gallery_thumbnail_path = '11_demos/_static/_thumb/demoSplitter.png'


class SplitterDemo(ItomUi):  # ListWidgetDemo is inherited from ItomUi
    def __init__(self):  # constructor
        # call constructor of ItomUi like it would be the constructor of the class itom.ui:
        ItomUi.__init__(self, "splitterDemo.ui", ui.TYPEWINDOW)

        self.gui.splitter.call("setStretchFactor", 0, 3)  # -> special method call
        self.gui.splitter.call("setStretchFactor", 1, 1)  # -> special method call
        self.gui.splitter.call("setCollapsible", 0, False)  # -> special method call
        self.gui.splitter.call("setCollapsible", 1, True)  # -> special method call
        print(
            "Top section. Collapsible:",
            self.gui.splitter.call("isCollapsible", 0),
        )  # -> special method call
        print(
            "Bottom section. Collapsible:",
            self.gui.splitter.call("isCollapsible", 1),
        )  # -> special method call

        # it is also possible to set the sizes of the sections using
        # self.gui.splitter.call("setSizes", (200, 300)) #-> special method call

        # or read it
        # print(self.gui.splitter.call("sizes")) #-> special method call


# create a first instance of ListWidgetDemo and the gui
win1 = SplitterDemo()
win1.gui.show()  # show the gui
