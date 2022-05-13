"""Multiple plots in horizontal layout
======================================

Example for dynamically changing the content of a form layout.
"""

from itomUi import (
    ItomUi,
)  # import the base class ItomUi from the module itomUi in the itom-packages subfolder
from itom import ui
from itom import uiItem
from itom import dataObject

# sphinx_gallery_thumbnail_path = '11_demos/_static/_thumb/demoMultipledPlotsLayout.png'


class MultiPlotHorLayout(ItomUi):
    def __init__(self):  # constructor

        # call constructor of ItomUi like it would be the constructor of the class itom.ui:
        ItomUi.__init__(self, "multiplePlotsInHorizontalLayout.ui", ui.TYPEWINDOW)

        # the spacing between each item of the layout is a property
        self.layout["spacing"] = 7

        # contents margins is left, top, right, bottom
        self.layout.call("setContentsMargins", 30, 15, 20, 5)

    @property
    def layout(self):
        """The reference to the horizontal layout."""
        return self.gui.horLayout

    @property
    def numWidgets(self):
        """Returns number of widgets in horLayout."""
        return self.layout.call("count")

    @ItomUi.autoslot("")
    def on_btnInfo_clicked(self):
        text = f"Num plots: {self.numWidgets}"

        for idx in range(self.numWidgets):
            widget: uiItem = self.layout.call("itemAt", idx)
            stretch: int = self.layout.call("stretch", idx)
            text += f"\nWidget {idx+1}: {str(widget)}, stretch: {stretch}"

        ui.msgInformation("Information", text, parent=self.gui)

    @ItomUi.autoslot("")
    def on_btnAddButton_clicked(self):
        className = "QPushButton"
        objectName = f"Button_{self.numWidgets}"
        obj: uiItem = self.layout.call("addItem", className, objectName)
        obj["text"] = objectName

        obj.connect("clicked()", self._buttonClicked)

        self.gui.btnRemove["enabled"] = self.numWidgets > 0
        self.gui.btnSetStretch["enabled"] = self.numWidgets > 0

    @ItomUi.autoslot("")
    def on_btnAddPlot_clicked(self):
        className = "itom2dqwtplot"
        objectName = f"Plot_{self.numWidgets}"
        obj: uiItem = self.layout.call("addItem", className, objectName)
        obj["source"] = dataObject.randN([30, 10])

        self.gui.btnRemove["enabled"] = self.numWidgets > 0
        self.gui.btnSetStretch["enabled"] = self.numWidgets > 0

    @ItomUi.autoslot("")
    def on_btnInsertButton_clicked(self):

        idx, valid = ui.getInt(
            "Position",
            "At which index should the button be inserted (-1: end)?",
            0,
            parent=self.gui,
        )

        if valid:
            className = "QPushButton"
            objectName = f"Button_{self.numWidgets}"
            obj: uiItem = self.layout.call("insertItem", idx, className, objectName)
            obj["text"] = objectName

            obj.connect("clicked()", self._buttonClicked)

            self.gui.btnRemove["enabled"] = self.numWidgets > 0
            self.gui.btnSetStretch["enabled"] = self.numWidgets > 0

    @ItomUi.autoslot("")
    def on_btnInsertFromUiFile_clicked(self):
        idx, valid = ui.getInt(
            "Position",
            "At which index should the widget from the UI file 'container.ui' be inserted (-1: end)?",
            0,
            parent=self.gui,
        )

        if valid:
            obj: uiItem = self.layout.call(
                "insertItemFromUiFile",
                idx,  # index
                "container.ui",  # filename to ui file
                "_%i" % self.numWidgets,  # prefix, added to the objectNames of all new widgets and layouts
            )

            self.gui.btnRemove["enabled"] = self.numWidgets > 0
            self.gui.btnSetStretch["enabled"] = self.numWidgets > 0

    @ItomUi.autoslot("")
    def on_btnRemove_clicked(self):

        if self.numWidgets <= 0:
            return

        labels = [self.layout.call("itemAt", idx)["objectName"] for idx in range(self.numWidgets)]

        name, valid = ui.getItem(
            "Widget to remove",
            "Select the widget to be removed",
            labels,
            editable=False,
        )

        if valid:
            idx = labels.index(name)
            self.layout.call("removeItemAt", idx)

        self.gui.btnRemove["enabled"] = self.numWidgets > 0
        self.gui.btnSetStretch["enabled"] = self.numWidgets > 0

    @ItomUi.autoslot("")
    def on_btnSetStretch_clicked(self):

        if self.numWidgets <= 0:
            return

        stretchs = [str(self.layout.call("stretch", idx)) for idx in range(self.numWidgets)]

        text, valid = ui.getText(
            "Stretch",
            f"Indicate a comma-separated list of stretch " f"factors for up to {self.numWidgets} widgets",
            ",".join(stretchs),
        )

        if valid:
            stretchs = text.split(",")

            if len(stretchs) > self.numWidgets:
                ui.msgCritical(
                    "Wrong input",
                    f"Stretchs must be a comma separated list of " f"integers (max. {self.numWidgets} entries)",
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

                self.layout.call("setStretch", idx, val)

    def _buttonClicked(self):
        # slot, called if any button is clicked
        ui.msgInformation("Button clicked", "The button has been clicked", parent=self.gui)


# create a first instance of AutoConnectExample and the gui
win1 = MultiPlotHorLayout()
win1.gui.show()  # show the gui


###############################################################################
# .. image:: ../../_static/demoMultiplePlotsLayout_1.png
#    :width: 100%
