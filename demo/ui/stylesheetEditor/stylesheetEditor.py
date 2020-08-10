# coding=iso-8859-15

"""GUI for live coding of a Qt stylesheet document.

This GUI can be used to live adjust a qss stylesheet
and directly see the changes for many widgets (Qt and itom
specific).

Usage:

Execute this script to load the demo GUI. This GUI consists
of a stylesheet editor widget in a left toolbar and many
widgets, that are often used within itom.

If you stylesheet depends on icons within a rcc resource file,
click the "load resources" button to load icons from a rcc
resource file, first. Then paste your desired stylesheet into
the editor. Whenever you want to update the GUI click the update button.
"""


from itomUi import ItomUi  # import the base class ItomUi from the module itomUi in the itom-packages subfolder
from itom import ui, dataObject

class StyleCreator(ItomUi): #StyleCreator is inherited from ItomUi
    
    def __init__(self): # constructor
        
        # call constructor of ItomUi like it would be the constructor of the class itom.ui:
        ItomUi.__init__(self, "stylecreator.ui", ui.TYPEWINDOW)
        
        # initialize some plots
        self.gui.itom1DQwtPlot["source"] = dataObject.rand([4, 500], 'uint16')
        
        self.gui.itom2dQwtPlot["source"] = dataObject.randN([100, 512, 768], 'uint8')
        self.gui.itom2dQwtPlot["overlayImage"] = dataObject.rand([512, 768], 'float32')
        self.gui.itom2dQwtPlot_2["source"] = dataObject.randN([100, 512, 768], 'uint8')
        self.gui.itom2dQwtPlot_2["overlayImage"] = dataObject.randN([512, 768], 'float32')
        
        self.gui.call("statusBar").call("showMessage","Status bar message...")
        
        self.on_btnUpdate_clicked()
        
    @ItomUi.autoslot("")  # the signal is clicked()
    def on_btnUpdate_clicked(self):
        # apply the current content of the text field to the stylesheet
        self.gui["styleSheet"] = self.gui.txtStyle["plainText"]
    
    @ItomUi.autoslot("")
    def on_btnLoadRcc_clicked(self):
        rccFile = ui.getOpenFileName("RCC Resource File",
                                     filters="RCC Resource File (*.rcc)",
                                     parent=self.gui)
        
        if rccFile is not None:
            if not registerResource(rccFile):
                ui.msgCritical("Error", f"Could not load resource file '{rccFile}'")


hmi = StyleCreator()
hmi.gui.show()

print("Text from python command line")
raise RuntimeError("Error text from python command line")
itom.clc()