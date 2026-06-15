"""Dynamic grid layout
======================

Example for dynamically changing the content of a grid layout.
"""

from itomUi import (
    ItomUi,
)  # import the base class ItomUi from the module itomUi in the itom-packages subfolder
from itom import ui

# sphinx_gallery_thumbnail_path = '11_demos/_static/_thumb/demoDynamicGridLayout.png'


class DynamicGridLayout(ItomUi):

    LOAD_FROM_UI = "<load from ui file>"

    def __init__(self):
        """Constructor.
        """

        # call constructor of ItomUi like it would be the constructor of the class itom.ui:
        ItomUi.__init__(self, "dynamicGridLayout.ui", ui.TYPEWINDOW)

        # the spacing between each item of the layout is a property
        self.layout["spacing"] = 7

        # contents margins is left, top, right, bottom
        self.layout.call("setContentsMargins", 30, 15, 20, 5)

        self.gui.comboAddWidget.call(
            "addItems",
            [
                self.LOAD_FROM_UI,
            ]
            + ui.availableWidgets(),
        )

        self._update()

    @property
    def layout(self):
        """The reference to the grid layout."""
        return self.gui.gridLayout

    @property
    def rowCount(self):
        """Returns number of rows in the grid layout."""
        return self.layout.call("rowCount")

    @property
    def columnCount(self):
        """Returns number of column in the grid layout."""
        return self.layout.call("columnCount")

    @ItomUi.autoslot("")
    def on_btnAddWidget_clicked(self):
        rowFrom = self.gui.spinAddRowFrom["value"]
        colFrom = self.gui.spinAddColFrom["value"]
        rowSpan = self.gui.spinAddRowSpan["value"]
        colSpan = self.gui.spinAddColSpan["value"]
        widget = self.gui.comboAddWidget["currentText"]

        if widget == self.LOAD_FROM_UI:
            filename = ui.getOpenFileName("UI File", filters="UI Files (*.ui)", parent=self.gui)

            if filename is not None:
                self.layout.call(
                    "addItemToGridFromUiFile",
                    filename,
                    f"_{rowFrom}_{colFrom}",
                    rowFrom,
                    colFrom,
                    rowSpan,
                    colSpan,
                )
        else:
            self.layout.call(
                "addItemToGrid",
                widget,
                f"item_{rowFrom}_{colFrom}",
                rowFrom,
                colFrom,
                rowSpan,
                colSpan,
            )

        self._update()

    @ItomUi.autoslot("")
    def on_btnRemoveWidget_clicked(self):
        row = self.gui.spinRemoveRow["value"]
        column = self.gui.spinRemoveColumn["value"]

        try:
            self.layout.call("removeItemFromGrid", row, column)
        except RuntimeError as ex:
            ui.msgCritical("Error", str(ex), parent=self.gui)

        self._update()

    @ItomUi.autoslot("")
    def on_btnInfo_clicked(self):
        text = f"Current grid size: {self.rowCount} rows x {self.columnCount} columns."
        text += "\n\n"

        for r in range(self.rowCount):
            for c in range(self.columnCount):
                try:
                    item = self.layout.call("itemAtPosition", r, c)
                except RuntimeError:
                    item = "-"
                text += f"Row {r}, col {c}: {item}\n"

        ui.msgInformation("Grid Content", text, parent=self.gui)

    @ItomUi.autoslot("")
    def on_btnColStretch_clicked(self):

        if self.rowCount * self.columnCount <= 0:
            return

        stretchs = [str(self.layout.call("columnStretch", idx)) for idx in range(self.columnCount)]

        text, valid = ui.getText(
            "Stretch",
            f"Indicate a comma-separated list of stretch " f"factors for up to {self.columnCount} columns",
            ",".join(stretchs),
        )

        if valid:
            stretchs = text.split(",")

            if len(stretchs) > self.columnCount:
                ui.msgCritical(
                    "Wrong input",
                    f"Stretchs must be a comma separated list of " f"integers (max. {self.columnCount} entries)",
                    parent=self.gui,
                )
                return

            for idx in range(len(stretchs)):
                try:
                    val = int(stretchs[idx])
                except ValueError:
                    ui.msgCiritcal(
                        "Wrong input",
                        f"Value '{stretchs[idx]}' is no integer number",
                        parent=self.gui,
                    )
                    return

                self.layout.call("setColumnStretch", idx, val)

    @ItomUi.autoslot("")
    def on_btnRowStretch_clicked(self):

        if self.rowCount * self.columnCount <= 0:
            return

        stretchs = [str(self.layout.call("rowStretch", idx)) for idx in range(self.rowCount)]

        text, valid = ui.getText(
            "Stretch",
            f"Indicate a comma-separated list of stretch " f"factors for up to {self.rowCount} columns",
            ",".join(stretchs),
        )

        if valid:
            stretchs = text.split(",")

            if len(stretchs) > self.rowCount:
                ui.msgCritical(
                    "Wrong input",
                    f"Stretchs must be a comma separated list of " f"integers (max. {self.rowCount} entries)",
                    parent=self.gui,
                )
                return

            for idx in range(len(stretchs)):
                try:
                    val = int(stretchs[idx])
                except ValueError:
                    ui.msgCiritcal(
                        "Wrong input",
                        f"Value '{stretchs[idx]}' is no integer number",
                        parent=self.gui,
                    )
                    return

                self.layout.call("setRowStretch", idx, val)

    def _update(self):
        self.gui.btnRemoveWidget["enabled"] = (self.rowCount * self.columnCount) > 0
        self.gui.btnColStretch["enabled"] = (self.rowCount * self.columnCount) > 0
        self.gui.btnRowStretch["enabled"] = (self.rowCount * self.columnCount) > 0
        self.gui.lblCaption["text"] = (
            f"Grid Layout (Current grid size: " f"{self.rowCount} rows x {self.columnCount} columns)"
        )


if __name__ == "__main__":
    # create a first instance of AutoConnectExample and the gui
    win1 = DynamicGridLayout()
    win1.gui.show()  # show the gui

###############################################################################
# .. image:: ../../_static/demoDynamicGridLayout_1.png
#    :width: 100%
