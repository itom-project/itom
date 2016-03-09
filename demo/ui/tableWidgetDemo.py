from itomUi import ItomUi #import the base class ItomUi from the module itomUi in the itom-packages subfolder
from itom import ui

class TableWidgetDemo(ItomUi):
    
    def __init__(self): #constructor
        ItomUi.__init__(self, "tableWidgetDemo.ui", ui.TYPEWINDOW)
        
    @ItomUi.autoslot("") #the signal is clicked()
    def on_btnSetValues_clicked(self):
        for c in range(0,3):
            for r in range(0,3):
                self.gui.table.call("setItem", r, c, "row %i, col %i" % (r, c))
        self.gui.table.call("resizeColumnsToContents")
        
    @ItomUi.autoslot("") #the signal is clicked()
    def on_btnSetHeaders_clicked(self):
        self.gui.table.call("setHorizontalHeaderLabels", ("label 1", "label 2", "label 3"))
        self.gui.table.call("setVerticalHeaderLabels", ("text 1", "text 2", "text 3"))
        
    @ItomUi.autoslot("") #the signal is clicked()
    def on_btnGetStatus_clicked(self):
        currentRow = self.gui.table.call("currentRow")
        currentColumn = self.gui.table.call("currentColumn")
        currentText = self.gui.table.call("getItem", currentRow, currentColumn)
        ui.msgInformation("Status", "Row: %i, Col: %i, Text: %s" % (currentRow, currentColumn, currentText), parent = self.gui)
    
    @ItomUi.autoslot("bool")
    def on_checkReadOnly_clicked(self, value):
        if not value:
            self.gui.table["editTriggers"] = "DoubleClicked;EditKeyPressed;AnyKeyPressed"
        else:
            self.gui.table["editTriggers"] = 0
            
    @ItomUi.autoslot("int,int")
    def on_table_cellClicked(self, row, column):
        self.gui.call("statusBar").call("showMessage","Cell %i,%i clicked" % (row, column),1000)
        
win1 = TableWidgetDemo()
win1.gui.show() #show the gui

