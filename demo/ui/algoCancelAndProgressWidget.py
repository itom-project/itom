from itomUi import ItomUi #import the base class ItomUi from the module itomUi in the itom-packages subfolder
from itom import ui
from itom import progressObserver

class AlgoCancelAndProgressWidget(ItomUi):
    
    def __init__(self): #constructor
        
        #call constructor of ItomUi like it would be the constructor of the class itom.ui:
        ItomUi.__init__(self, "algoCancelAndProgressWidget.ui", ui.TYPEWINDOW)
        
        self.observer = progressObserver(progressBar = self.gui.progressBar, \
                                         label = self.gui.lblProgress, \
                                         progressMinimum = 0, \
                                         progressMaximum = 100)
        
    
        
#create a first instance of AlgoCancelAndProgressWidget and the gui
win1 = AlgoCancelAndProgressWidget()
win1.gui.show() #show the gui
