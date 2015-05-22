# coding=iso-8859-15

from itom import *
from itomUi import *
import inspect
import os
import sys

reloadModules = 1

class EmbeddedPlots(ItomUi):
    def __init__(self, systemPath = None):
        self.startDir = os.getcwd()
        self.upDating = True
        self.measureType = 0
        
        if(systemPath is None):
            ownFilename = inspect.getfile(inspect.currentframe())
            ownDir = os.path.dirname(os.path.realpath(__file__)) #os.path.dirname(ownFilename)
        else:
            ownDir = systemPath
        
        uiFile = os.path.join(ownDir, "embedded2DwLinePlot.ui")
        uiFile = os.path.abspath(uiFile)
        ItomUi.__init__(self, uiFile, ui.TYPEWINDOW, childOfMainWindow=True)
        
        try:
            self.gui.twoDPlot["staticLineCutID"] = self.gui.oneDPlot
        except:
            pass
        self.gui.oneDPlot["visible"] = False
        
    def init(self, newTopo, newIntensity = None):
        
        self.rawTopo = newTopo.copy()
        
        self.gui.twoDPlot["source"] = newTopo
        
        if newIntensity is not None and newIntensity.dims > 1:
            self.gui.twoDPlot["overlayImage"] = newIntensity
        
        self.upDating = False
        
    def show(self,modalLevel = 0):
        ret = self.gui.show(modalLevel)
    
    def setColorMap(self,colorMap):
        self.gui.twoDPlot["colorMap"] = colorMap
        #self.gui.threeDPlot["colorMap"] = colorMap
    

if(__name__ == '__main__'):
    prot = EmbeddedPlots(None)
    try:
        topo
        prot.init(topo, intensity)
    except:
        tempInt = dataObject.randN([5, 100, 100], 'uint16')
        tempData = dataObject.randN([5, 100, 100], 'float32') * 0.002
        for i in range(0, tempInt.shape[0]):
            tempData[i, :, :] += i - tempInt.shape[0] / 2
            tempData[i, :, 50:100] += tempInt.shape[0] / 4
        prot.init(tempData, tempInt)
    prot.show()