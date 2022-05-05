"""List widget
==============

This demo shows how to use the auto-connection feature
for automatically connecting signals from widgets to methods.

The base requirement for this is, that the ui-file is wrapped
by a class in Python.

.. hint::
    This demo uses specially wrapped methods of QListWidget. For more information see
    section 'Calling slots' in https://itom.bitbucket.io/latest/docs/06_extended_gui/qtdesigner.html)
    
    These methods are indiciated by #-> special method call
"""

from itomUi import (
    ItomUi,
)  # import the base class ItomUi from the module itomUi in the itom-packages subfolder
from itom import ui
# sphinx_gallery_thumbnail_path = '11_demos/_static/_thumb/demoListWidget.png'


class ListWidgetDemo(ItomUi):  # ListWidgetDemo is inherited from ItomUi
    def __init__(self):  # constructor
        # call constructor of ItomUi like it would be the constructor of the class itom.ui:
        ItomUi.__init__(self, "listWidgetDemo.ui", ui.TYPEWINDOW)

        # from now on, you can use the member self.gui to access the handle of the user interface
        self.on_btnAddItems_clicked()

    @ItomUi.autoslot("")  # the signal is clicked()
    def on_btnAddItems_clicked(self):
        count = self.gui.listMain["count"]
        size = 3
        newItemTexts = ["item %i" % i for i in range(count, count + size)]

        self.gui.listMain.call(
            "addItems", newItemTexts
        )  # -> special method call

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

        # set flags of all new items
        for i in range(count, count + size):

            if i % 2 == 0:
                self.gui.listMain.call(
                    "setFlags", i, flag1
                )  # -> special method call
                self.gui.listMain.call(
                    "setCheckState", i, checked
                )  # -> special method call
            else:
                self.gui.listMain.call(
                    "setFlags", i, flag2
                )  # -> special method call
                self.gui.listMain.call(
                    "setCheckState", i, partially
                )  # -> special method call

    @ItomUi.autoslot("")  # the signal is clicked()
    def on_btnClearAll_clicked(self):
        self.gui.listMain.call("clear")

    @ItomUi.autoslot("")  # connect to clicked() signal of btnEval
    def on_btnEval_clicked(self):
        count = self.gui.listMain["count"]
        for i in range(count):
            itemText = self.gui.listMain.call(
                "item", i
            )  # -> special method call
            itemFlags = self.gui.listMain.call(
                "flags", i
            )  # -> special method call
            checkState = self.gui.listMain.call(
                "checkState", i
            )  # -> special method call
            checkStateStr = ["unchecked", "partially", "checked"][checkState]
            print(
                "Item %i: %s, flags: %i, check state: %s"
                % (i, itemText, itemFlags, checkStateStr)
            )

    @ItomUi.autoslot("int")  # the signal is currentRowChanged(int)
    def on_listMain_currentRowChanged(self, row):
        print("current row changed to row:", row)


# create a first instance of ListWidgetDemo and the gui
win1 = ListWidgetDemo()
win1.gui.show()  # show the gui
