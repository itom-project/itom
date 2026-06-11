"""Dynamic form layout
======================

Example for dynamically changing the content of a form layout.

This example shows how new widgets are added or inserted to a form
layout in an user interface, how entire rows are removed or how
the existing widgets in a form layout are requested.

Most functions used in this example are based on special methods
of the form layout (Qt class: ``QFormLayout``), that are made accessible
in itom via the ``uiItem.call('methodName', *args)`` method.
"""

from itomUi import ItomUi
from itom import ui
from itom import uiItem
# sphinx_gallery_thumbnail_path = '11_demos/_static/_thumb/demoDynamicFormLayout.png'


class DynamicFormLayout(ItomUi):
    """Main class with the functionality of the user interface."""

    def __init__(self):
        """Constructor."""
        # call constructor of ItomUi like it would be the constructor of the class itom.ui:
        ItomUi.__init__(self, "dynamicFormLayout.ui", ui.TYPEWINDOW)

        # the spacing between each item of the layout is a property
        self.layout["spacing"] = 7

        self.gui.comboAddWidget.call("addItems", ui.availableWidgets())

        self._update()

    @property
    def layout(self):
        """The reference to the form layout."""
        return self.gui.formLayout

    @property
    def rowCount(self):
        """Returns number of rows in the form layout."""
        return self.layout.call("rowCount")

    @ItomUi.autoslot("")
    def on_btnAddRowLineEdit_clicked(self):
        """Adds a new row with a string label and
        a line edit widget (class name in Qt: QLineEdit."""
        num = self.rowCount
        name = f"LineEdit_{num+1}"
        objName = name
        self.layout.call("addRow", name, "QLineEdit", objName)
        self._update()

    @ItomUi.autoslot("")
    def on_btnAddRowCheckBox_clicked(self):
        """Adds a new row with a string label and
        a line edit widget (class name in Qt: QCheckBox."""
        num = self.rowCount
        name = f"CheckBox_{num+1}"
        objName = name
        box = self.layout.call("addRow", name, "QCheckBox", objName)
        box["checked"] = True
        self._update()

    @ItomUi.autoslot("")
    def on_btnInsertRowComboBox_clicked(self):
        row, valid = ui.getInt(
            "Row index",
            "At which row should the combo box be inserted?",
            0,
            parent=self.gui,
        )

        if valid:
            name = f"ComboBox_{self.rowCount+1}"
            objName = name
            combo = self.layout.call("insertRow", row, name, "QComboBox", objName)
            combo.call("addItems", ("Option 1", "Option 2", "Option 3"))
            self._update()

    @ItomUi.autoslot("")
    def on_btnSetWidget_clicked(self):
        rowIndex = self.gui.spinAddRowIndex["value"]

        # 0: modify label, 1: modify field, 2: widget spans both columns
        role = self.gui.comboAddRole["currentIndex"]

        className = self.gui.comboAddWidget["currentText"]

        self.layout.call(
            "setItem", rowIndex, role, className, f"item_{rowIndex}_{role}"
        )
        self._update()

    @ItomUi.autoslot("")
    def on_btnRemoveRow_clicked(self):
        row, valid = ui.getInt(
            "Row index",
            "Which row index should be removed?",
            0,
            min=0,
            max=self.rowCount - 1,
            parent=self.gui,
        )

        if valid:
            self.layout.call("removeRow", row)
            self._update()

    @ItomUi.autoslot("")
    def on_btnInfo_clicked(self):
        items = []
        label: uiItem
        field: uiItem

        for r in range(self.rowCount):
            try:
                label = self.layout.call("itemAtPosition", r, 0)  # 0 stands for label
                className = label.getClassName()

                if className == "QLabel":
                    lblStr = label["text"]
                else:
                    lblStr = str(label)

            except RuntimeError:
                lblStr = "<no label>"

            try:
                field = self.layout.call("itemAtPosition", r, 1)  # 1 stands for field
                className = field.getClassName()

                if className == "QLineEdit":
                    fieldStr = field["text"]
                elif className == "QCheckBox":
                    fieldStr = "Check Box"
                elif className == "QComboBox":
                    fieldStr = field["currentText"]
                else:
                    fieldStr = str(field)
            except RuntimeError:
                fieldStr = "<no field>"

            items.append(f"{r + 1}: {lblStr}: {fieldStr}")

        text = (
            f"Form Layout with {self.rowCount} rows:\n"
            + "-----------------------------------------\n\n"
            + "\n".join(items)
        )

        ui.msgInformation("Info", text, parent=self.gui)

    def _update(self):
        self.gui.btnRemoveRow["enabled"] = self.rowCount > 0
        self.gui.lblCaption["text"] = f"Form Layout (Currently {self.rowCount} rows):"


if __name__ == "__main__":
    # create an object of DynamicFormLayout and shows it
    win1 = DynamicFormLayout()
    win1.gui.show()

###############################################################################
# .. image:: ../../_static/demoDynamicFormLayout_1.png
#    :width: 75%
