"""ItomUI class
===============

Demo to show how to create a user interface in an object oriented approach.

This demo uses the base class ``ItomUi`` from the module ``itomUi``. This module
is distributed with itom and can directly be imported (located in the itom-packages
directory).

The proposal is to create one class, that represents all code, that is related
to one ui file. This class is therefore derived from ``ItomUi``.

The benefits are:

1. It is possible to create member methods in your class, that are automatically
   called if the user does a specific action in the user interface (e.g. clicks
   a button). For this, the method needs to have a specific method name. Then,
   this method is considered to be a ``slot`` and is automatically connected with
   the ``signal``, that is emitted due to the action. Methods, that should be
   auto-connected slots must be decorated with the @ItomUi.autoslot decorator.

2. Use the factory function ``ItomUi.disableGui`` for with statements to
   switch some items of the GUI to a specific state (disable, enable, hide, show)
   before starting a long operation and automatically switch back to the previous
   state at the end of the operation. Even if the operation fails with an unhandled
   exception, the GUI is reverted to the original state.
"""

from itomUi import ItomUi
from itom import ui
import time

# sphinx_gallery_thumbnail_path = '11_demos/_static/_thumb/demoItomUI.png'


class ItomUiClassDemo(ItomUi):
    """This class contains the code for the user interface "itomUiClassDemo.ui"."""

    def __init__(self):
        """Constructor.

        Instead of directly creating an object of the class ``itom.ui``
        call the constructor of the base class with the same arguments than
        ``itom.ui``. The ui object is then accessible by the member ``self.gui``.

        Use the constructor to further initialize the user interface.
        """
        ItomUi.__init__(
            self, "itomUiClassDemo.ui", type=ui.TYPEWINDOW, deleteOnClose=True
        )

        # further initialization
        self.gui.progressBar["visible"] = False

    @ItomUi.autoslot("bool")
    def on_checkEnabled_toggled(self, checked):
        """Auto-slot if the check box has been checked or unchecked.

        This method is automatically connected with the ``toggled`` signal of
        the check box with the object name ``checkEnabled`` (see QtDesigner >>
        property toolbox) since it has the ``ItomUi.autoslot`` decorator and since
        the method name follows the pattern ``on_<objectName>_<signal>``.

        The signal is the original C++ signal name of the widget (item). It can
        be obtained either from the Qt documentation or by calling the ``info`` method
        of the ``itom.uiItem``, that represents the desired widget. If the signal
        has some arguments, write a comma separated list (without separators) of the
        C++ datatypes of the arguments in the string argument of the decorator.

        Args:
            checked (bool): This argument is the boolean argument of
                the signal ``toggled(bool checked)`` of the checkbox (class QCheckBox).
        """
        self.gui.groupBox["enabled"] = checked

    @ItomUi.autoslot("")  # the signal is clicked()
    def on_btnStart_clicked(self):
        """Auto-slot that is connected with the ``clicked`` signal of the button btnStart.

        This method emulates a long operation. During the operation, the
        group box is disabled and the progress bar of the ui is shown.
        Afterwards, both properties are reverted to their original state
        (dependent on the current enabled state of the group box).
        """
        # the following items will be disabled (enabled=False) during with statement
        disableItems = [
            self.gui.groupBox,
            self.gui.checkEnabled,
            self.gui.btnStart,
        ]

        # the following items will be shown (visible=True) during with statement
        showItems = [
            self.gui.progressBar,
        ]

        # the following items will be hidden (visible=False) during with statement
        hideItems = [
            self.gui.btnStartException,
        ]

        # the following items will be enabled (enabled=True) during with statement
        enableItems = []

        with self.disableGui(
            disableItems=disableItems,
            showItems=showItems,
            hideItems=hideItems,
            enableItems=enableItems,
            revertToInitialStateOnExit=True,
            showWaitCursor=self.gui.checkWaitCursor["checked"],
        ):
            # long going operation within the with statement
            time.sleep(3)

    @ItomUi.autoslot("")  # the signal is clicked()
    def on_btnStartException_clicked(self):
        """Auto-slot that is connected with the ``clicked`` signal of the button btnStartException.

        This method emulates a long operation. During the operation, the
        group box is disabled and the progress bar of the ui is shown.
        Afterwards, both properties are reverted to their original state
        (dependent on the current enabled state of the group box).

        The special thing of this demo is, that an unhandled exception happens
        during the operation. Nevertheless, the GUI is turned into its previous
        state. This is a feature of the ``disableGui`` factory function for
        a with statement.
        """
        # the following items will be disabled (enabled=False) during with statement
        disableItems = [
            self.gui.groupBox,
            self.gui.checkEnabled,
            self.gui.btnStartException,
        ]

        # the following items will be shown (visible=True) during with statement
        showItems = [
            self.gui.progressBar,
        ]

        # the following items will be hidden (visible=False) during with statement
        hideItems = [
            self.gui.btnStart,
        ]

        # the following items will be enabled (enabled=True) during with statement
        enableItems = []

        with self.disableGui(
            disableItems=disableItems,
            showItems=showItems,
            hideItems=hideItems,
            enableItems=enableItems,
            revertToInitialStateOnExit=True,
            showWaitCursor=self.gui.checkWaitCursor["checked"],
        ):
            # long going operation within the with statement
            time.sleep(3)

            raise RuntimeError(
                "an unhandled exception occurred, but the "
                "GUI is turned into the original state though"
            )


if __name__ == "__main__":
    win = ItomUiClassDemo()
    win.show()  # this method internally calls ``itom.ui.show``.
