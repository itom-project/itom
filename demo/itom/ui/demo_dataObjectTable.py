# coding=utf8
"""Dataobject table
===================

"""
from itom import ui
from itom import dataObject
# sphinx_gallery_thumbnail_path = '11_demos/_static/_thumb/demoTableWidget.png'

def printContent():
    gui.table["data"].data()


def printInfo():
    gui.table.info()


def cellClicked(row, column):
    # see statusbar example for details about accessing the statusBar of a QMainWindow
    gui.call("statusBar").call(
        "showMessage",
        "cell clicked. row:" + str(row) + ", col:" + str(column),
        1000,
    )


def radioUInt8Clicked():
    gui.table["data"] = dataObject.randN([10, 20], "uint8")
    gui.spinDecimals["enabled"] = False
    gui.spinDecimalsEditing["enabled"] = False
    gui.table["horizontalResizeMode"] = "ResizeToContents"


def radioFloat32Clicked():
    gui.table["data"] = dataObject.randN([2, 2], "float32")
    gui.spinDecimals["enabled"] = True
    gui.spinDecimalsEditing["enabled"] = True
    gui.table["horizontalResizeMode"] = "Stretch"


def radioComplex64Clicked():
    gui.table["data"] = dataObject.randN([3, 4], "complex64")
    gui.spinDecimals["enabled"] = True
    gui.spinDecimalsEditing["enabled"] = True
    gui.table["horizontalResizeMode"] = "Stretch"


def spinDecimalsChanged(val):
    gui.table["decimals"] = val


def spinDecimalsEditingChanged(val):
    gui.table["editorDecimals"] = val


def checkReadonlyChanged(val):
    gui.table["readOnly"] = val


def comboLabelsChanged(idx):
    if idx == 0:
        gui.table["suffixes"] = ()
        gui.table["horizontalLabels"] = ()
        gui.table["verticalLabels"] = ()
        gui.table["horizontalHeaderVisible"] = True
        gui.table["verticalHeaderVisible"] = True
    elif idx == 1:
        gui.table["suffixes"] = (" mm",)
        gui.table["horizontalLabels"] = ("row 1", "row 2", "row 3")
        gui.table["verticalLabels"] = ("col 1", "col 2", "col 3", "col 4")
        gui.table["horizontalHeaderVisible"] = True
        gui.table["verticalHeaderVisible"] = True
    elif idx == 2:
        gui.table["suffixes"] = (" mm", u" \xb0", "")
        gui.table["horizontalHeaderVisible"] = False
        gui.table["verticalHeaderVisible"] = False


gui = ui("dataObjectTableDemo.ui", ui.TYPEWINDOW, deleteOnClose=True)
radioUInt8Clicked()
gui.spinDecimals["value"] = gui.table["decimals"]
gui.spinDecimalsEditing["value"] = gui.table["editorDecimals"]

gui.cmdPrint.connect("clicked()", printContent)
gui.cmdPrintInfo.connect("clicked()", printInfo)
gui.radioUInt8.connect("clicked()", radioUInt8Clicked)
gui.radioFloat32.connect("clicked()", radioFloat32Clicked)
gui.radioComplex64.connect("clicked()", radioComplex64Clicked)
gui.spinDecimals.connect("valueChanged(int)", spinDecimalsChanged)
gui.spinDecimalsEditing.connect("valueChanged(int)", spinDecimalsEditingChanged)
gui.checkReadonly.connect("toggled(bool)", checkReadonlyChanged)
gui.comboLabels.connect("currentIndexChanged(int)", comboLabelsChanged)
gui.table.connect("clicked(int,int)", cellClicked)

gui.show()

###############################################################################
# .. image:: ../../_static/demoDataObjectTable_1.png
#    :width: 100%