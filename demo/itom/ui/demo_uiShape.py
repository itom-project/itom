"""UI shape
===========

This demo shows a window with a 2D plot as center widget.
A random dataObject is shown in the plot. The user can then
create various geometric shapes (like points, lines, rectangles...)
in the plot either by the toolbar of the plot or by pressing dedicated buttons.

Additionally, many signals of the plot are connected to python slots
to show what kind of information can for instance be obtained by the plot
upon the creation of plots. There are still more signals available, however this
demo shows a base set already. Furthermore, it is possible to force the user to
create new shapes either in a modal or non-modal process. The first might be
interesting if a script requires the input of a region in a plot before continuing with
the script execution"""

from itom import dataObject
from itom import ui
from itomUi import ItomUi
from itom import shape
import inspect
import os.path

# sphinx_gallery_thumbnail_path = '11_demos/_static/_thumb/demoUIShape.png'


class UiShapeDemo(ItomUi):
    def __init__(self):
        ownFilename = inspect.getfile(inspect.currentframe())
        ownDir = os.path.dirname(ownFilename)
        uiFile = os.path.join(ownDir, "uiShapeDemo.ui")
        uiFile = os.path.abspath(uiFile)
        ItomUi.__init__(self, uiFile, ui.TYPEWINDOW, childOfMainWindow=True)

        dObj = dataObject.randN([100, 300], "uint8")
        dObj.axisScales = (0.2, 0.2)
        dObj.axisUnits = ("mm", "mm")
        self.gui.plot["source"] = dObj

    def show(self, modalLevel=0):
        self.gui.show(modalLevel)

    def drawShapeButtonsEnable(self, enabled, modal=False):
        self.gui.btnCancel["enabled"] = not enabled and not modal
        self.gui.btnAddPoint["enabled"] = enabled
        self.gui.btnAddLine["enabled"] = enabled
        self.gui.btnAddRect["enabled"] = enabled
        self.gui.btnAddSquare["enabled"] = enabled
        self.gui.btnAddEllipse["enabled"] = enabled
        self.gui.btnAddCircle["enabled"] = enabled

    def startInteraction(self, type):
        nrOfElements = self.gui.spinNoToAdd["value"]
        modal = self.gui.checkModalInput["checked"]
        self.drawShapeButtonsEnable(False, modal)

        if not modal:
            self.gui.plot.call("userInteractionStart", type, True, nrOfElements)
        else:
            try:
                # drawAndPickElements throws RuntimeError if user interrupts the process to add a new shape or new shapes
                # drawAndPickElements is a memeber of itom.plotItem, self.gui.plot however is of its base type uiItem.
                # Therefore, the uiItem is cast to plotItem, first.
                shapes = plotItem(self.gui.plot).drawAndPickElements(type, nrOfElements)
                self.gui.listLog.call(
                    "addItem",
                    "End of modal interaction. %i shapes added:" % len(shapes),
                )
            except RuntimeError:
                self.gui.listLog.call("addItem", "Modal interaction interrupted")

    @ItomUi.autoslot("")
    def on_btnAddPoint_clicked(self):
        self.startInteraction(shape.Point)

    @ItomUi.autoslot("")
    def on_btnAddLine_clicked(self):
        self.startInteraction(shape.Line)

    @ItomUi.autoslot("")
    def on_btnAddRect_clicked(self):
        self.startInteraction(shape.Rectangle)

    @ItomUi.autoslot("")
    def on_btnAddSquare_clicked(self):
        self.startInteraction(shape.Square)

    @ItomUi.autoslot("")
    def on_btnAddEllipse_clicked(self):
        self.startInteraction(shape.Ellipse)

    @ItomUi.autoslot("")
    def on_btnAddCircle_clicked(self):
        self.startInteraction(shape.Circle)

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
    def on_btnClearSelected_clicked(self):
        self.gui.plot.call("deleteGeometricShape", self.gui.plot["selectedGeometricShape"])

    @ItomUi.autoslot("")
    def on_btnDrawShapes_clicked(self):
        self.gui.plot.call("addGeometricShape", shape(shape.Circle, (30, 10), 8))  # head
        self.gui.plot.call("addGeometricShape", shape(shape.Rectangle, (24, 15), (28, 12)))  # left eye
        self.gui.plot.call("addGeometricShape", shape(shape.Rectangle, (32, 15), (36, 12)))  # right eye
        self.gui.plot.call("addGeometricShape", shape(shape.Ellipse, (25, 5.25), (35, 4.75)))  # mouth

    @ItomUi.autoslot("")
    def on_plot_geometricShapesDeleted(self):
        self.gui.btnCreateAndShowMask["enabled"] = False
        self.gui.btnClearAll["enabled"] = False
        self.gui.listLog.call("addItem", "all shapes deleted")

    @ItomUi.autoslot("int,ito::Shape")
    def on_plot_geometricShapeAdded(self, idx, shape):
        self.gui.btnCreateAndShowMask["enabled"] = True
        self.gui.btnClearAll["enabled"] = True
        self.gui.listLog.call("addItem", "shape %i added: " % idx + str(shape))

    @ItomUi.autoslot("int,ito::Shape")
    def on_plot_geometricShapeChanged(self, idx, shape):
        self.gui.listLog.call("addItem", "shape %i changed: " % idx + str(shape))

    @ItomUi.autoslot("ito::Shape")
    def on_plot_geometricShapeCurrentChanged(self, shape):
        self.gui.btnClearSelected["enabled"] = shape.valid
        self.gui.call("statusBar").call("showMessage", "Current shape changed to: %s" % str(shape), 1000)

    @ItomUi.autoslot("QVector<ito::Shape>,bool")
    def on_plot_geometricShapeFinished(self, shapes, aborted):
        self.drawShapeButtonsEnable(True)
        if not aborted:
            self.gui.listLog.call(
                "addItem",
                "successfully finished to add or change the following shapes: " + str(shapes),
            )
        else:
            self.gui.listLog.call(
                "addItem",
                "adding geometric shape(s) aborted. %i shape(s) already added" % len(shapes),
            )
        self.gui.listLog.call("scrollToBottom")

    @ItomUi.autoslot("int,bool")
    def on_plot_geometricShapeStartUserInput(self, type, userInteractionReason):
        if userInteractionReason == False:  # user selected a button in the toolbar to draw a new shape, disable buttons
            self.drawShapeButtonsEnable(False)

    @ItomUi.autoslot("")
    def on_btnClearLog_clicked(self):
        self.gui.listLog.call("clear")

    @ItomUi.autoslot("")
    def on_btnCancel_clicked(self):
        self.gui.plot.call("userInteractionStart", -1, False, 0)


if __name__ == "__main__":
    dObj = dataObject.randN([600, 800], "float32")
    uiShapeDemo = UiShapeDemo()
    uiShapeDemo.show()

###############################################################################
# .. image:: ../../_static/demoUIShape_1.png
#    :width: 100%
