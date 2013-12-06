from itom import dataObject
from itom import ui
from itomUi import ItomUi
import inspect
import os.path
import numpy as np

reloadModules = 1

class measureToolMain(ItomUi):
    
    upDating = True
    measureType = 0
    plotElementType = 0
    
    elementCount = 0
    
    enumGeomNames = ["noType", "point", "line", "rectangle", "square", "ellipse", "circle", "polygon"]
    enumGeomIdx = [0, 101, 102, 103, 104, 105, 106, 110]
    enumRelationName = None
    enumRelationIdx = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    
    def __init__(self):
        self.upDating = True
        self.measureType = 0
        ownFilename = inspect.getfile(inspect.currentframe())
        ownDir = os.path.dirname(ownFilename)
        uiFile = os.path.join(ownDir, "uiMeasureToolMain.ui")
        uiFile = os.path.abspath(uiFile)
        ItomUi.__init__(self, uiFile, ui.TYPEDIALOG, childOfMainWindow=True)
        
        self.enumRelationName = list(self.gui.measurementTable["relationNames"])
        #self.enumRelationName.append('Sa value')
        #self.enumRelationName.append('Sz10 value')
        #self.enumRelationName.append('Sq value')
        #self.enumRelationName.append('step height')
        
        #self.gui.measurementTable["relationNames"] = self.enumRelationName
        self.gui.measurementTable.call('addRelationName', 'mean value')
        
        self.enumRelationName = self.gui.measurementTable["relationNames"]
        
        self.enumRelationIdx[0] = self.enumRelationName.index('N.A.')
        self.enumRelationIdx[1] = self.enumRelationName.index('radius (own)')
        self.enumRelationIdx[2] = self.enumRelationName.index('distance to')
        self.enumRelationIdx[3] = self.enumRelationName.index('length (own)')
        self.enumRelationIdx[4] = self.enumRelationName.index('area')
        self.enumRelationIdx[5] = self.enumRelationName.index('mean value')
        
        self.elementCount = 0
        
    def init(self):
        ret = self.gui.dataPlot["source"] = dataObject.zeros([1,1])
        self.upDating = False
    
    def show(self,modalLevel = 0):
        ret = self.gui.show(modalLevel)
    
    def showObject(self, newObject = None):
        
        if(self.measureType!= 0):
            self.on_pushButtonCancel_clicked()
        else:
            self.gui.measurementTable.call("clearAll")
            self.gui.dataPlot.call("clearGeometricElements")
            self.elementCount = self.gui.dataPlot["geometricElementsCount"]
            
        if(newObject is None):
            self.gui.dataPlot["source"] = dataObject.zeros([1,1])
        else:
            self.gui.dataPlot["source"] = newObject
    
    def clearPlots(self):
        ret = self.gui.dataPlot["source"] = dataObject.zeros([1,1])
        self.gui.measurementTable.call("clearAll")
        self.gui.dataPlot.call("clearGeometricElements")
        self.elementCount = self.gui.dataPlot["geometricElementsCount"]
    
    @ItomUi.autoslot("")
    def on_pushButtonDistanceP2P_clicked(self):
        if(self.measureType== 0):
            self.measureType = 2
            self.plotElementTyp = self.enumGeomIdx[self.enumGeomNames.index('point')]
            self.elementCount = self.gui.dataPlot["geometricElementsCount"]
            self.gui.dataPlot.call("userInteractionStart", self.plotElementTyp, True, 2)
        
    
    @ItomUi.autoslot("")
    def on_pushButtonDistanceP2L_clicked(self):
        if(self.measureType== 0):
            self.measureType = 3
            self.plotElementTyp = self.enumGeomIdx[self.enumGeomNames.index('point')]
            self.elementCount = self.gui.dataPlot["geometricElementsCount"]
            self.gui.dataPlot.call("userInteractionStart", self.plotElementTyp, True, 1)
        
    @ItomUi.autoslot("")
    def on_pushButtonRadius_clicked(self):
        if(self.measureType== 0):
            self.measureType = 1
            self.plotElementTyp = self.enumGeomIdx[self.enumGeomNames.index('ellipse')]
            self.elementCount = self.gui.dataPlot["geometricElementsCount"]
            self.gui.dataPlot.call("userInteractionStart", self.plotElementTyp, True, 1)
    
    @ItomUi.autoslot("")
    def on_pushButtonMean_clicked(self):
        if(self.measureType== 0):
            self.measureType = 4
            self.plotElementTyp = self.enumGeomIdx[self.enumGeomNames.index('rectangle')]
            self.elementCount = self.gui.dataPlot["geometricElementsCount"]
            self.gui.dataPlot.call("userInteractionStart", self.plotElementTyp, True, 1)
    
    @ItomUi.autoslot("")
    def on_pushButtonCancel_clicked(self):
        if(self.measureType!= 0):
            self.measureType = 0
            self.gui.dataPlot.call("userInteractionStart", self.plotElementTyp, False, 0)
            self.plotElementTyp = 0
        
    @ItomUi.autoslot("")
    def on_pushButtonClearAll_clicked(self):
        if(self.measureType!= 0):
            self.measureType = 0
            self.gui.dataPlot.call("userInteractionStart", self.plotElementTyp, False, 0)
            self.plotElementTyp = 0
        
        self.gui.measurementTable.call("clearAll")
        self.gui.dataPlot.call("clearGeometricElements")
        self.elementCount = self.gui.dataPlot["geometricElementsCount"]
        
#    @ItomUi.autoslot("int")
#    def on_dataPlot_plotItemChanged(self, index):
#        geometricElements = self.gui.dataPlot["geometricElements"]
#        self.gui.measurementTable["source"] = geometricElements
    
    @ItomUi.autoslot("int, bool")
    def on_dataPlot_plotItemsFinished(self, type, aborted):
        geometricElements = self.gui.dataPlot["geometricElements"]
        self.gui.measurementTable["source"] = geometricElements
        
        newElementCount = self.gui.dataPlot["geometricElementsCount"]
        
        first = 0.0
        second = -1.0
        
        if(self.elementCount + 1 > newElementCount):
            self.measureType = 0
            return
        
        if(self.measureType == 1):    #radius
            first = geometricElements[newElementCount-1, 0]
            self.gui.measurementTable.call("addRelation" , dataObject(np.array([first, self.enumRelationIdx[1], -1.0, 0.0])))
            
        elif(self.measureType == 2):    #distance point to point P2P
            if(self.elementCount + 2 <= newElementCount):
                first = geometricElements[newElementCount-2, 0]
                second = geometricElements[newElementCount-1, 0]
            else:
                first = geometricElements[newElementCount-1, 0]
            
            self.gui.measurementTable.call("addRelation" , dataObject(np.array([first, self.enumRelationIdx[2], second, 0.0])))
            
        elif(self.measureType == 3):    #distance point to point P2L the first time, still missing a line
            self.measureType = 33
            self.gui.measurementTable.call("addRelation" , dataObject(np.array([first, self.enumRelationIdx[2], second, 0.0])))
            self.plotElementTyp = self.enumGeomIdx[self.enumGeomNames.index('line')]
            self.gui.dataPlot.call("userInteractionStart", self.plotElementTyp, True, 1)
            return
        
        elif(self.measureType == 33):    #distance point to point P2L the second time, now having a line
            if(self.elementCount + 2 <= newElementCount):
                first = geometricElements[newElementCount-2, 0]
                second = geometricElements[newElementCount-1, 0]
            
                try:
                    relToEdit = self.gui.measurementTable["lastAddedRelation"]
                    self.gui.measurementTable.call("modifyRelation" , relToEdit, dataObject(np.array([first, self.enumRelationIdx[2], second, 0.0])))
                except:
                    print("setting second geometric element failed")
            else:
                print("setting second geometric element failed")
        
        elif(self.measureType == 4):    #mean-Value
            first = geometricElements[newElementCount-1, 0]
            
            try:
                
                tempObj = self.gui.dataPlot["source"]
                
                tempScales = tempObj.axisScales
                tempOffsets = tempObj.axisOffsets 
                val0 = int(geometricElements[newElementCount-1, 2] * (1 / tempScales[1]) - tempOffsets[1])
                val1 = int(geometricElements[newElementCount-1, 5] * (1 / tempScales[1]) - tempOffsets[1])
                x0 = min(val0, val1)
                x1 = max(val0, val1) + 1
                
                x0 = max(x0, 0)
                x1 = max(x1, 0)
                x0 = min(x0, tempObj.shape[tempObj.dims-1] - 1)
                x1 = min(x1, tempObj.shape[tempObj.dims-1])
                
                val0 = int(geometricElements[newElementCount-1, 3] * (1 / tempScales[0]) - tempOffsets[0])
                val1 = int(geometricElements[newElementCount-1, 6] * (1 / tempScales[0]) - tempOffsets[0])
                y0 = min(val0, val1)
                y1 = max(val0, val1) + 1
                
                y0 = max(y0, 0)
                y1 = max(y1, 0)
                y0 = min(y0, tempObj.shape[tempObj.dims-2] - 1)
                y1 = min(y1, tempObj.shape[tempObj.dims-2])
                
                meanValue = filter("meanValue", tempObj[y0:y1, x0:x1])
                
            except:
                meanValue = np.NaN
                
            self.gui.measurementTable.call("addRelation" , dataObject(np.array([first, self.enumRelationIdx[4], -1.0, 0.0])))
            self.gui.measurementTable.call("addRelation" , dataObject(np.array([first, self.enumRelationIdx[5] + 0x8000, -1.0, meanValue])))
            
            
        self.measureType = 0
        self.elementCount = newElementCount
    
if(__name__ == '__main__'):
    dObj = dataObject.randN([600, 800], 'float32')
    measurementTool = measureToolMain()
    measurementTool.show()
    dObj.axisScales = (0.2, 0.2)
    dObj.axisUnits = ('mm', 'mm')
    measurementTool.showObject(dObj)
    