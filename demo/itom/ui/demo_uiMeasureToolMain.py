"""Measure tool
===============

"""

from itom import dataObject
from itom import ui
from itomUi import ItomUi
from itom import shape
import inspect
import os.path
import numpy as np
# sphinx_gallery_thumbnail_path = '11_demos/_static/_thumb/demoMeasureTool.png'


class MeasureToolMain(ItomUi):
    upDating = True
    measureType = 0
    plotElementType = 0

    elementCount = 0

    enumGeomNames = [
        "noType",
        "point",
        "line",
        "rectangle",
        "square",
        "ellipse",
        "circle",
        "polygon",
    ]
    enumGeomIdx = [
        shape.Invalid,
        shape.Point,
        shape.Line,
        shape.Rectangle,
        shape.Square,
        shape.Ellipse,
        shape.Circle,
        shape.Polygon,
    ]
    enumRelationName = None
    enumRelationIdx = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    def __init__(self):
        self.upDating = True
        self.measureType = 0
        ownFilename = inspect.getfile(inspect.currentframe())
        ownDir = os.path.dirname(ownFilename)
        uiFile = os.path.join(ownDir, "uiMeasureToolMain.ui")
        uiFile = os.path.abspath(uiFile)
        ItomUi.__init__(self, uiFile, ui.TYPEWINDOW, childOfMainWindow=True)

        self.gui.measurementTable.call("addRelationName", "mean value")
        self.enumRelationName = self.gui.measurementTable["relationNames"]
        self.enumRelationIdx[0] = self.enumRelationName.index("N.A.")
        self.enumRelationIdx[1] = self.enumRelationName.index("radius (own)")
        self.enumRelationIdx[2] = self.enumRelationName.index("distance to")
        self.enumRelationIdx[3] = self.enumRelationName.index("length (own)")
        self.enumRelationIdx[4] = self.enumRelationName.index("area")
        self.enumRelationIdx[5] = self.enumRelationName.index("mean value")

        self.elementCount = 0

    def show(self, modalLevel=0):
        ret = self.gui.show(modalLevel)

    def showObject(self, newObject=None):
        if self.measureType != 0:
            self.on_pushButtonCancel_clicked()
        else:
            self.gui.measurementTable.call("clearAll")
            self.gui.dataPlot.call("clearGeometricShapes")
            self.elementCount = self.gui.dataPlot["geometricShapesCount"]

        if newObject is None:
            self.gui.dataPlot["source"] = dataObject.zeros([1, 1])
        else:
            self.gui.dataPlot["source"] = newObject

    def clearPlots(self):
        ret = self.gui.dataPlot["source"] = dataObject.zeros([1, 1])
        self.gui.measurementTable.call("clearAll")
        self.gui.dataPlot.call("clearGeometricShapes")
        self.elementCount = self.gui.dataPlot["geometricShapesCount"]

    @ItomUi.autoslot("")
    def on_pushButtonDistanceP2P_clicked(self):
        if self.measureType == 0:
            self.measureType = 2
            self.plotElementTyp = shape.Point
            self.elementCount = self.gui.dataPlot["geometricShapesCount"]
            self.gui.pushButtonCancel["enabled"] = True
            self.gui.dataPlot.call("userInteractionStart", self.plotElementTyp, True, 2)

    @ItomUi.autoslot("")
    def on_pushButtonDistanceP2L_clicked(self):
        if self.measureType == 0:
            self.measureType = 3
            self.plotElementTyp = shape.Point
            self.elementCount = self.gui.dataPlot["geometricShapesCount"]
            self.gui.pushButtonCancel["enabled"] = True
            self.gui.dataPlot.call("userInteractionStart", self.plotElementTyp, True, 1)

    @ItomUi.autoslot("")
    def on_pushButtonRadius_clicked(self):
        if self.measureType == 0:
            self.measureType = 1
            self.plotElementTyp = shape.Ellipse
            self.elementCount = self.gui.dataPlot["geometricShapesCount"]
            self.gui.pushButtonCancel["enabled"] = True
            self.gui.dataPlot.call("userInteractionStart", self.plotElementTyp, True, 1)

    @ItomUi.autoslot("")
    def on_pushButtonMean_clicked(self):
        if self.measureType == 0:
            self.measureType = 4
            self.plotElementTyp = shape.Rectangle
            self.elementCount = self.gui.dataPlot["geometricShapesCount"]
            self.gui.pushButtonCancel["enabled"] = True
            self.gui.dataPlot.call("userInteractionStart", self.plotElementTyp, True, 1)

    @ItomUi.autoslot("")
    def on_pushButtonCancel_clicked(self):
        if self.measureType != 0:
            self.measureType = 0
            self.gui.dataPlot.call(
                "userInteractionStart", self.plotElementTyp, False, 0
            )
            self.plotElementTyp = 0
            self.gui.pushButtonCancel["enabled"] = False

    @ItomUi.autoslot("")
    def on_pushButtonClearAll_clicked(self):
        if self.measureType != 0:
            self.measureType = 0
            self.gui.dataPlot.call(
                "userInteractionStart", self.plotElementTyp, False, 0
            )
            self.plotElementTyp = 0

        self.gui.measurementTable.call("clearAll")
        self.gui.dataPlot.call("clearGeometricShapes")
        self.elementCount = self.gui.dataPlot["geometricShapesCount"]

    #    @ItomUi.autoslot("int")
    #    def on_dataPlot_plotItemChanged(self, index):
    #        geometricElements = self.gui.dataPlot["geometricElements"]
    #        self.gui.measurementTable["source"] = geometricElements
    @ItomUi.autoslot("")
    def on_dataPlot_geometricShapesDeleted(self):
        self.gui.pushButtonClearAll["enabled"] = False

    @ItomUi.autoslot("int,ito::Shape")
    def on_dataPlot_geometricShapeAdded(self, idx, shape):
        self.gui.pushButtonClearAll["enabled"] = True

    @ItomUi.autoslot("QVector<ito::Shape>,bool")
    def on_dataPlot_geometricShapeFinished(self, shapes, aborted):
        geometricShapes = self.gui.dataPlot["geometricShapes"]
        self.gui.measurementTable["geometricShapes"] = geometricShapes

        newElementCount = self.gui.dataPlot["geometricShapesCount"]

        if self.elementCount + 1 > newElementCount:
            self.measureType = 0
            return

        if self.measureType == 1:  # radius
            first = geometricShapes[newElementCount - 1]
            self.gui.measurementTable.call(
                "addRelation",
                dataObject(np.array([first.index, self.enumRelationIdx[1], -1, 0.0])),
            )

        elif self.measureType == 2:  # distance point to point P2P
            if self.elementCount + 2 <= newElementCount:
                first = geometricShapes[newElementCount - 2]
                second = geometricShapes[newElementCount - 1]
            else:
                first = geometricShapes[newElementCount - 1]

            self.gui.measurementTable.call(
                "addRelation",
                dataObject(
                    np.array(
                        [
                            first.index,
                            self.enumRelationIdx[2],
                            second.index,
                            0.0,
                        ]
                    )
                ),
            )

        elif (
            self.measureType == 3
        ):  # distance point to point P2L the first time, still missing a line
            self.measureType = 33
            first = geometricShapes[newElementCount - 1]
            self.gui.measurementTable.call(
                "addRelation",
                dataObject(np.array([first.index, self.enumRelationIdx[2], -1, 0.0])),
            )
            self.plotElementTyp = self.enumGeomIdx[self.enumGeomNames.index("line")]
            self.gui.dataPlot.call("userInteractionStart", self.plotElementTyp, True, 1)
            return

        elif (
            self.measureType == 33
        ):  # distance point to point P2L the second time, now having a line
            if self.elementCount + 2 <= newElementCount:
                first = geometricShapes[newElementCount - 2]
                second = geometricShapes[newElementCount - 1]

                try:
                    relToEdit = self.gui.measurementTable["lastAddedRelation"]
                    self.gui.measurementTable.call(
                        "modifyRelation",
                        relToEdit,
                        dataObject(
                            np.array(
                                [
                                    first.index,
                                    self.enumRelationIdx[2],
                                    second.index,
                                    0.0,
                                ]
                            )
                        ),
                    )
                except:
                    print("setting second geometric element failed")
            else:
                print("setting second geometric element failed")

        elif self.measureType == 4:  # mean-Value
            first = geometricShapes[newElementCount - 1]
            try:
                tempObj = self.gui.dataPlot["source"]

                rect = geometricShapes[newElementCount - 1]
                x0 = round(tempObj.physToPix(min(rect.point1[0], rect.point2[0]), 1))
                x1 = (
                    round(tempObj.physToPix(max(rect.point1[0], rect.point2[0]), 1)) + 1
                )
                x0 = max(x0, 0)
                x1 = max(x1, 0)
                x0 = min(x0, tempObj.shape[tempObj.dims - 1] - 1)
                x1 = min(x1, tempObj.shape[tempObj.dims - 1])

                y0 = round(tempObj.physToPix(min(rect.point1[1], rect.point2[1]), 1))
                y1 = (
                    round(tempObj.physToPix(max(rect.point1[1], rect.point2[1]), 1)) + 1
                )
                y0 = max(y0, 0)
                y1 = max(y1, 0)
                y0 = min(y0, tempObj.shape[tempObj.dims - 2] - 1)
                y1 = min(y1, tempObj.shape[tempObj.dims - 2])

                meanValue = filter("meanValue", tempObj[y0:y1, x0:x1])

            except:
                meanValue = np.NaN

            self.gui.measurementTable.call(
                "addRelation",
                dataObject(np.array([first.index, self.enumRelationIdx[4], -1, 0.0])),
            )
            self.gui.measurementTable.call(
                "addRelation",
                dataObject(
                    np.array(
                        [
                            first.index,
                            self.enumRelationIdx[5] + 0x8000,
                            -1,
                            meanValue,
                        ]
                    )
                ),
            )

        self.measureType = 0
        self.elementCount = newElementCount
        self.gui.pushButtonCancel["enabled"] = False


if __name__ == "__main__":
    dObj = dataObject.randN([600, 800], "float32")
    measurementTool = MeasureToolMain()
    measurementTool.show()
    dObj.axisScales = (0.2, 0.2)
    dObj.axisUnits = ("mm", "mm")
    measurementTool.showObject(dObj)
