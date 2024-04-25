"""Algorithm cancel and progress widget
=======================================

This script shows how the ``itom.progressObserver`` is used
to observe and report the progress of functions.
"""

from itomUi import (
    ItomUi,
)  # import the base class ItomUi from the module itomUi in the itom-packages subfolder
from itom import ui
from itom import uiItem
from itom import progressObserver
from contextlib import contextmanager
from typing import Dict
# sphinx_gallery_thumbnail_path = '11_demos/_static/_thumb/demoAlgoCancelAndProgress.png'


class AlgoCancelAndProgressWidget(ItomUi):
    def __init__(self):  # constructor
        # call constructor of ItomUi like it would be the constructor of the class itom.ui:
        ItomUi.__init__(self, "algoCancelAndProgressWidget.ui", ui.TYPEWINDOW)

        self.observer = progressObserver(
            progressBar=self.gui.progressBar,
            label=self.gui.lblProgress,
            progressMinimum=0,
            progressMaximum=100,
        )

        self.gui.btnCancel["visible"] = False
        self.gui.lblProgress["visible"] = False
        self.gui.progressBar["visible"] = False
        self.gui.btnCancel.invokeProgressObserverCancellation(
            "clicked()", self.observer
        )

    @ItomUi.autoslot("")
    def on_btnStart_clicked(self):
        with self.disableGui(
            {
                self.gui.btnStart: False,
                self.gui.btnCancel: True,
                self.gui.lblProgress: True,
                self.gui.progressBar: True,
            }
        ):
            # the following filter must have the ability to provide status information (see information of filter)
            filter("demoCancellationFunction", _observer=self.observer)

    @contextmanager
    def disableGui(self, widgets: Dict[uiItem, bool]):
        """this is a smart helper method that can be used in a with context.
        It changes the visible property when entering the with context for
        all given uiItems to the given boolean value, then executes the content
        of the with statement, and finally switches the visible properties back
        to the origin.The switch back is executed even if an exception (cancellation
        of the algorithm etc.) occurred.
        """
        for w in widgets:
            w["visible"] = widgets[w]
        try:
            yield
        finally:
            for w in widgets:
                w["visible"] = not widgets[w]


# create a first instance of AlgoCancelAndProgressWidget and the gui
win1 = AlgoCancelAndProgressWidget()
win1.gui.show()  # show the gui
