# coding=iso-8859-15

from itom import *
from itomUi import *
import inspect
import os
import sys

reloadModules = 1

class EmbeddedPlots(ItomUi):
    def __init__(self, systemPath = None, interpreteAsZ = False):
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
        
        if interpreteAsZ:
            self.gui.twoDPlot["zSlicePlotItem"] = self.gui.oneDPlot
        else:
            self.gui.twoDPlot["lineCutPlotItem"] = self.gui.oneDPlot
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
    modeID = 1 # use 0 for lateral slice example or 1 for zSlice example
    prot = EmbeddedPlots(None, modeID is 1)
    try:
        topo
        prot.init(topo, intensity)
    except:
        
        if modeID == 1:
            tempData = dataObject.randN([10, 50, 50], 'float32')
            for i in range(0, tempData.shape[0]):
                tempData[i, :, :] += i - tempData.shape[0] / 20
                tempData[i, :, 25:50] += tempData.shape[0] / 4
        else:
            tempData = dataObject.randN([3, 100, 100], 'float32') * 0.002
            for i in range(0, tempData.shape[0]):
                tempData[i, :, :] += i - tempData.shape[0] / 2
                tempData[i, :, 50:100] += tempData.shape[0] / 4
        prot.init(tempData)
    prot.show()