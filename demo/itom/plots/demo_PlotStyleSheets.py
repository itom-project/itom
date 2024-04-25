"""Plot style sheets
====================

This demo shows how you can set and change the theme of your GUI.
The style can not be entirely removed to the windows, mac or plastique style
(usually done by ``setStyle(new QWindowsStyle())``).
It only resets the style sheet to an empty one, such that the native fallbacks to the
os dependent style is applied. However, ``Qt::WA_StyledBackground`` is still active.
"""

from itom import dataObject
from itom import ui
from itomUi import ItomUi
import inspect
import os.path
import numpy as np
# sphinx_gallery_thumbnail_path = '11_demos/_static/_thumb/demoPlotStyleSheet.png'


class PlotStyleSheets(ItomUi):
    def __init__(self):
        ownFilename = inspect.getfile(inspect.currentframe())
        ownDir = os.path.dirname(ownFilename)
        uiFile = os.path.join(ownDir, "demoPlotStyleSheets.ui")
        uiFile = os.path.abspath(uiFile)
        ItomUi.__init__(self, uiFile, ui.TYPEWINDOW, childOfMainWindow=True)
        obj = dataObject.randN([1024, 1024], "float32")
        obj[200:300, 200:300] = float("nan")
        self.gui.plot2d["source"] = obj
        self.gui.plot1d["source"] = dataObject(
            np.sin(np.arange(0, 10 * np.pi, (1 / 20) * np.pi))
        )

    def show(self, modalLevel: int = 0):
        self.gui.show(modalLevel)

    @ItomUi.autoslot("")
    def on_radioNoStyle_clicked(self):
        self.gui["styleSheet"] = ""
        self.gui.plot2d["backgroundColor"] = "#ffffff"
        self.gui.plot2d["axisColor"] = "#000000"
        self.gui.plot2d["textColor"] = "#000000"
        self.gui.plot2d["canvasColor"] = "#ffffff"
        self.gui.plot1d["backgroundColor"] = "#ffffff"
        self.gui.plot1d["axisColor"] = "#000000"
        self.gui.plot1d["textColor"] = "#000000"
        self.gui.plot1d["canvasColor"] = "#ffffff"

    @ItomUi.autoslot("")
    def on_radioDarkStyle_clicked(self):
        with open("darkorange.qss") as f:
            self.gui["styleSheet"] = f.read()
        self.gui.plot2d["backgroundColor"] = "#323232"
        self.gui.plot2d["axisColor"] = "#ffffff"
        self.gui.plot2d["textColor"] = "#ffffff"
        self.gui.plot2d["canvasColor"] = "#323232"
        self.gui.plot1d["backgroundColor"] = "#323232"
        self.gui.plot1d["axisColor"] = "#ffffff"
        self.gui.plot1d["textColor"] = "#ffffff"
        self.gui.plot1d["canvasColor"] = "#323232"

    @ItomUi.autoslot("")
    def on_radioButtonBright_clicked(self):
        self.gui.plot2d["buttonSet"] = "StyleBright"
        self.gui.plot1d["buttonSet"] = "StyleBright"

    @ItomUi.autoslot("")
    def on_radioButtonDark_clicked(self):
        self.gui.plot2d["buttonSet"] = "StyleDark"
        self.gui.plot1d["buttonSet"] = "StyleDark"


if __name__ == "__main__":
    instance = PlotStyleSheets()
    instance.show()

###############################################################################
# .. image:: ../../_static/demoPlotStyleSheets_1.png
#    :width: 100%

###############################################################################
# .. image:: ../../_static/demoPlotStyleSheets_2.png
#    :width: 100%
