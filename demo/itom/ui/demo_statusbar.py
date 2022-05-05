"""Statusbar
============

This is a short tutorial about how to use the statusbar
"""

from itomUi import ItomUi
from itom import ui
import os
import inspect

# sphinx_gallery_thumbnail_path = '11_demos/_static/_thumb/demoStatusBar.png'


class Statusbar(ItomUi):
    def __init__(self):
        """get current path and and initialize the GUI"""
        dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
        absPath = os.path.join(dir, "statusbar.ui")
        ItomUi.__init__(
            self,
            absPath,
            ui.TYPEWINDOW,
            ui.BUTTONBAR_VERTICAL,
            {"AcceptRole": "OK", "RejectRole": "Cancel"},
        )

    @ItomUi.autoslot("")
    def on_btnAdd_clicked(self):
        """call the status bar and show the message: Here I am"""
        self.gui.call("statusBar").call("showMessage", "Here I am")

    @ItomUi.autoslot("")
    def on_btnShow_clicked(self):
        """call the status bar and show the message: I am here for a second. The message will disappear after 1000 ms"""
        self.gui.call("statusBar").call("showMessage", "I am here for a second", 1000)

    @ItomUi.autoslot("")
    def on_btnDelete_clicked(self):
        """clear the status bar"""
        self.gui.call("statusBar").call("clearMessage")


if __name__ == "__main__":

    inst = Statusbar()
    inst.show()  # show the gui