"""Auto connect signals
=======================

This demo shows how to use the auto-connection feature
for automatically connecting signals from widgets to methods.

The base requirement for this is, that the ui-file is wrapped
by a class in Python."""

from itomUi import (
    ItomUi,
)  # import the base class ItomUi from the module itomUi in the itom-packages subfolder
from itom import ui
# sphinx_gallery_thumbnail_path = '11_demos/_static/_thumb/demoAutoConnect.png'


class AutoConnectExample(ItomUi):  # AutoConnectExample is inherited from ItomUi
    def __init__(self):  # constructor
        # call constructor of ItomUi like it would be the constructor of the class itom.ui:
        ItomUi.__init__(self, "autoConnectDemo.ui", ui.TYPEWINDOW)
        self.counter = 0  # create a counter variable for this instance

        # initialize the captions of the labels:
        self.gui.lblCheckResult["text"] = "not checked"
        self.gui.lblSpinResult["text"] = "current value: 0"

    @ItomUi.autoslot("")  # the signal is clicked()
    def on_btnDemo_clicked(self):
        # increment the counter
        self.counter += 1
        ui.msgInformation(
            "button clicked",
            "The button was clicked %i times" % self.counter,
            parent=self.gui,
        )
        self.gui.btnDemo["text"] = "click me again"

    @ItomUi.autoslot("bool")  # the signal is clicked(bool checked)
    def on_checkDemo_clicked(self, checked):
        if checked:
            self.gui.lblCheckResult["text"] = "checked"
        else:
            self.gui.lblCheckResult["text"] = "not checked"

    @ItomUi.autoslot("int")  # the signal is valueChanged ( int i )
    def on_spinDemo_valueChanged(self, value):
        self.gui.lblSpinResult["text"] = "current value: %i" % value


# create a first instance of AutoConnectExample and the gui
win1 = AutoConnectExample()
win1.gui.show()  # show the gui
win1.gui["geometry"] = (100, 100, 412, 157)

# create a second instance (due to the class based approach, both windows have different counter variables (among others)
win2 = AutoConnectExample()
win2.gui.show()  # show the gui
