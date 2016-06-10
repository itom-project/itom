from itom import dataObject
from itom import ui
from itomUi import ItomUi
from itom import shape
import inspect
import os.path
import numpy as np

class UiShapeDemo(ItomUi):
    
    def __init__(self):
        ownFilename = inspect.getfile(inspect.currentframe())
        ownDir = os.path.dirname(ownFilename)
        uiFile = os.path.join(ownDir, "uiShapeDemo.ui")
        uiFile = os.path.abspath(uiFile)
        ItomUi.__init__(self, uiFile, ui.TYPEWINDOW, childOfMainWindow=True)
        
        dObj = dataObject.randN([100,300],'uint8')
        dObj.axisScales = (0.2, 0.2)
        dObj.axisUnits = ('mm', 'mm')
        self.gui.plot["source"] = dObj
    
    def show(self,modalLevel = 0):
        ret = self.gui.show(modalLevel)
    
    @ItomUi.autoslot("")
    def on_btnAddPoint_clicked(self):
        nrOfElements = self.gui.spinNoToAdd["value"]
        self.gui.plot.call("userInteractionStart", shape.Point, True, nrOfElements)
        self.gui.btnCancel["enabled"] = True
        
    @ItomUi.autoslot("")
    def on_btnAddLine_clicked(self):
        nrOfElements = self.gui.spinNoToAdd["value"]
        self.gui.plot.call("userInteractionStart", shape.Line, True, nrOfElements)
        self.gui.btnCancel["enabled"] = True
    
    @ItomUi.autoslot("")
    def on_btnAddRect_clicked(self):
        nrOfElements = self.gui.spinNoToAdd["value"]
        self.gui.plot.call("userInteractionStart", shape.Rectangle, True, nrOfElements)
        self.gui.btnCancel["enabled"] = True
        
    @ItomUi.autoslot("")
    def on_btnAddSquare_clicked(self):
        nrOfElements = self.gui.spinNoToAdd["value"]
        self.gui.plot.call("userInteractionStart", shape.Square, True, nrOfElements)
        self.gui.btnCancel["enabled"] = True
    
    @ItomUi.autoslot("")
    def on_btnAddEllipse_clicked(self):
        nrOfElements = self.gui.spinNoToAdd["value"]
        self.gui.plot.call("userInteractionStart", shape.Ellipse, True, nrOfElements)
        self.gui.btnCancel["enabled"] = True
    
    @ItomUi.autoslot("")
    def on_btnAddCircle_clicked(self):
        nrOfElements = self.gui.spinNoToAdd["value"]
        self.gui.plot.call("userInteractionStart", shape.Circle, True, nrOfElements)
        self.gui.btnCancel["enabled"] = True
    
    @ItomUi.autoslot("bool")
    def on_checkAllowToolbar_clicked(self, checked):
        self.gui.plot["geometricShapesDrawingEnabled"] = checked
    
    @ItomUi.autoslot("bool")
    def on_checkAllowMove_clicked(self, checked):
        modes = []
        flags = 0
        if self.gui.checkAllowMove["checked"]:
            modes.append("Move")
        else:
            flags |= shape.MoveLock
        if self.gui.checkAllowResize["checked"]:
            modes.append("Resize")
        else:
            flags |= shape.ResizeLock
        
        if len(modes) > 0:
            self.gui.plot["geometryModificationModes"] = ";".join(modes)
        else:
            self.gui.plot["geometryModificationModes"] = ""
        
        #modify the flags of all existing shapes
        shapes = self.gui.plot["geometricShapes"]
        for i in range(0, len(shapes)):
            shapes[i].flags = flags
        self.gui.plot["geometricShapes"] = shapes

    @ItomUi.autoslot("bool")
    def on_checkAllowResize_clicked(self, checked):
        self.on_checkAllowMove_clicked(checked)
    
    @ItomUi.autoslot("")
    def on_btnCreateAndShowMask_clicked(self):
        mask = self.gui.plot["source"].createMask(self.gui.plot["geometricShapes"])
        plot(mask)
    
    @ItomUi.autoslot("")
    def on_btnClearAll_clicked(self):
        self.gui.plot.call("clearGeometricShapes")
    
    @ItomUi.autoslot("")
    def on_plot_geometricShapesDeleted(self):
        self.gui.btnCreateAndShowMask["enabled"] = False
        self.gui.btnClearAll["enabled"] = False
    
    @ItomUi.autoslot("int,ito::Shape")
    def on_plot_geometricShapeAdded(self, idx, shape):
        self.gui.btnCreateAndShowMask["enabled"] = True
        self.gui.btnClearAll["enabled"] = True
    
if(__name__ == '__main__'):
    dObj = dataObject.randN([600, 800], 'float32')
    uiShapeDemo = UiShapeDemo()
    uiShapeDemo.show()
    
    