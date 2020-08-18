''''''

from itomUi import ItomUi #import the base class ItomUi from the module itomUi in the itom-packages subfolder
from itom import ui, uiItem, dataObject


class MultiPlotHorLayout(ItomUi):
    
    def __init__(self): #constructor
        
        #call constructor of ItomUi like it would be the constructor of the class itom.ui:
        ItomUi.__init__(self, "multiplePlotsInHorizontalLayout.ui", ui.TYPEWINDOW)
    
    @property
    def layout(self):
        """The reference to the horizontal layout."""
        return self.gui.horLayout
        
    @property
    def numWidgets(self):
        """Returns number of widgets in horLayout."""
        return self.layout.call("count")
        
    @ItomUi.autoslot("")
    def on_btnInfo_clicked(self):
        text = f"Num plots: {self.numWidgets}"
        
        ui.msgInformation("Information", text, parent=self.gui)
    
    @ItomUi.autoslot("")
    def on_btnAddButton_clicked(self):
        className = "QPushButton"
        objectName = f"Button_{self.numWidgets}"
        obj: uiItem = self.layout.call("addItem", className, objectName)
        obj["text"] = objectName
        
        obj.connect("clicked()", self._buttonClicked)
        
        self.gui.btnRemove["enabled"] = self.numWidgets > 0
    
    @ItomUi.autoslot("")
    def on_btnAddPlot_clicked(self):
        className = "itom2dqwtplot"
        objectName = f"Plot_{self.numWidgets}"
        obj: uiItem = self.layout.call("addItem", className, objectName)
        obj["source"] = dataObject.randN([30, 10])
        
        self.gui.btnRemove["enabled"] = self.numWidgets > 0
    
    @ItomUi.autoslot("")
    def on_btnRemove_clicked(self):
        self.gui.btnRemove["enabled"] = self.numWidgets > 0
    
    def _buttonClicked(self):
        ui.msgInformation("Button clicked", "The button has been clicked",
                          parent=self.gui)


# create a first instance of AutoConnectExample and the gui
win1 = MultiPlotHorLayout()
win1.gui.show()  # show the gui