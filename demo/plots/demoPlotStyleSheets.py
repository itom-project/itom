from itom import dataObject
from itom import ui
from itomUi import ItomUi
import inspect
import os.path
import numpy as np

class PlotStyleSheets(ItomUi):
    def __init__(self):
        ownFilename = inspect.getfile(inspect.currentframe())
        ownDir = os.path.dirname(ownFilename)
        uiFile = os.path.join(ownDir, "demoPlotStyleSheets.ui")
        uiFile = os.path.abspath(uiFile)
        ItomUi.__init__(self, uiFile, ui.TYPEWINDOW, childOfMainWindow=True)
        obj = dataObject.randN([1024,1024],'float32')
        obj[200:300,200:300] = float('nan')
        self.gui.plot2d["source"] = obj
        self.gui.plot1d["source"] = dataObject(np.sin(np.arange(0, 10 * np.pi, (1/20) * np.pi)))
        
    
    def show(self,modalLevel = 0):
        ret = self.gui.show(modalLevel)
        
    @ItomUi.autoslot("")
    def on_radioNoStyle_clicked(self):
        #up to now, the style can not be entirely removed to the windows, mac or plastique style
        #(usually done by setStyle(new QWindowsStyle()))
        #It only resets the style sheet to an empty one, such that the native fallbacks to the
        #os dependent style is applied. However, Qt::WA_StyledBackground is still active.
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
        with open("darkorange.qss", "rt") as f:
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

def userdemo_plotStyleSheets():
    instance = PlotStyleSheets()
    instance.show()

if __name__ == "__main__":
    instance = PlotStyleSheets()
    instance.show()