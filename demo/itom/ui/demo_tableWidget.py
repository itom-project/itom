"""Table widget
===============

"""
from itomUi import (
    ItomUi,
)  # import the base class ItomUi from the module itomUi in the itom-packages subfolder
from itom import ui
import random
# sphinx_gallery_thumbnail_path = '11_demos/_static/_thumb/demoTableWidget.png'


class TableWidgetDemo(ItomUi):
    def __init__(self):  # constructor
        ItomUi.__init__(self, "tableWidgetDemo.ui", ui.TYPEWINDOW)
        self.filled = False

    @ItomUi.autoslot("")  # the signal is clicked()
    def on_btnSetValues_clicked(self):
        for c in range(0, 3):
            for r in range(0, 3):
                self.gui.table.call("setItem", r, c, "row %i, col %i" % (r, c))
        self.gui.table.call("resizeColumnsToContents")
        self.filled = True

    @ItomUi.autoslot("")  # the signal is clicked()
    def on_btnSetHeaders_clicked(self):
        self.gui.table.call(
            "setHorizontalHeaderLabels", ("label 1", "label 2", "label 3")
        )
        self.gui.table.call(
            "setVerticalHeaderLabels", ("text 1", "text 2", "text 3")
        )

    @ItomUi.autoslot("")  # the signal is clicked()
    def on_btnGetStatus_clicked(self):
        currentRow = self.gui.table.call("currentRow")
        currentColumn = self.gui.table.call("currentColumn")
        currentText = self.gui.table.call("getItem", currentRow, currentColumn)
        ui.msgInformation(
            "Status",
            "Row: %i, Col: %i, Text: %s"
            % (currentRow, currentColumn, currentText),
            parent=self.gui,
        )

    @ItomUi.autoslot("bool")
    def on_checkReadOnly_clicked(self, value):
        if not value:
            self.gui.table[
                "editTriggers"
            ] = "DoubleClicked;EditKeyPressed;AnyKeyPressed"
        else:
            self.gui.table["editTriggers"] = 0

    @ItomUi.autoslot("int,int")
    def on_table_cellClicked(self, row, column):
        self.gui.call("statusBar").call(
            "showMessage", "Cell %i,%i clicked" % (row, column), 1000
        )

    @ItomUi.autoslot("int,int")
    def on_table_cellChanged(self, row, column):
        checkState = self.gui.table.call(
            "checkState", row, column
        )  # -> special method call
        checkStateStr = ["unchecked", "partially", "checked"][checkState]
        currentText = self.gui.table.call("getItem", row, column)
        print(
            "Cell %i,%i (%s) changed: %s"
            % (row, column, currentText, checkStateStr)
        )

    @ItomUi.autoslot("")  # the signal is clicked()
    def on_btnAddCheckboxes_clicked(self):

        if not self.filled:
            ui.msgCritical("empty table", "fill the content first")
            return

        # define the flags which parameterize every item in the list (individually, if desired)
        # the flag is an OR-combination of the enumeration Qt::ItemFlag
        flagSelectable = 1  # Qt::ItemIsSelectable
        flagCheckable = 16  # Qt::ItemIsUserCheckable
        flagEnabled = 32  # Qt::ItemIsEnabled
        flagTristate = 256  # Qt::ItemIsUserTristate
        flag1 = (
            flagSelectable | flagCheckable | flagEnabled
        )  # only checkable with on/off state
        flag2 = flag1 | flagTristate  # checkable with on/off/partially state

        # the check state is the state of the checkbox, according Qt::CheckState enumeration
        checked = 2  # checked
        partially = 1  # partially
        unchecked = 0  # unchecked

        # tuple of possible combinations between flags and check state
        flags = [
            [flag1, checked],
            [flag1, unchecked],
            [flag2, checked],
            [flag2, partially],
            [flag2, unchecked],
        ]

        for m in range(self.gui.table["rowCount"]):
            for n in range(self.gui.table["columnCount"]):
                f = flags[random.randint(0, len(flags) - 1)]
                self.gui.table.call(
                    "setFlags", m, n, f[0]
                )  # -> special method call
                self.gui.table.call(
                    "setCheckState", m, n, f[1]
                )  # -> special method call
        self.gui.table.call("resizeColumnsToContents")


win1 = TableWidgetDemo()
win1.gui.show()  # show the gui
