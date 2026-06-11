"""
Base class for custom user interfaces with auto-slot connections.

License information:

itom software
URL: http://www.uni-stuttgart.de/ito
Copyright (C) 2020, Institut für Technische Optik (ITO),
Universität Stuttgart, Germany

This file is part of itom.

itom is free software; you can redistribute it and/or modify it
under the terms of the GNU Library General Public License as published by
the Free Software Foundation; either version 2 of the License, or (at
your option) any later version.

itom is distributed in the hope that it will be useful, but
WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Library
General Public License for more details.

You should have received a copy of the GNU Library General Public License
along with itom. If not, see <http://www.gnu.org/licenses/>.
"""

from contextlib import contextmanager
from typing import List, Tuple, Union

import itom
from itom import ui

__version__ = "2.5.0"


class ItomUi:
    """Base class which can be inherited in order to show an user defined
    user-interface. This class provides possibilities for auto-connecting
    decorated methods in your implementation with certain signals of widgets
    in the user interface.

    Example:
    - User interface contains a button 'pushButton'
    - Write a method which should be connected to the
      buttons clicked(bool)-signal:

        @ItomUi.autoslot("bool")
        def on_pushButton_clicked(self,arg):
            # this method is auto-connected in the constructor of ItomUi
            pass

    - This step is the same than typing:
        self.ui.pushButton.connect("clicked(bool)", self.on_pushButton_clicked)
    """

    def __init__(
        self,
        filename: str,
        type: int = ui.TYPEWINDOW,
        dialogButtonBar: int = ui.BUTTONBAR_NO,
        dialogButtons: dict = None,
        childOfMainWindow: bool = True,
        deleteOnClose: bool = False,
        dockWidgetArea: int = ui.TOPDOCKWIDGETAREA,
        **kwds,
    ):
        """Constructor.

        Call this constructor from a derived class like you would
        directly call the constructor of the class ``itom.ui``.

        Args:
            filename (str): path to the user interface file (*.ui), absolute or relative
                to current directory.
            type (int): This ``type`` defines how the loaded user interface is
                displayed:

                * ``ui.TYPEDIALOG`` (0): The ui-file is the content of a dialog window
                  or, if the file already defines a `QDialog`, this dialog is shown as
                  it is. This is recommended for the creation of modal dialogs,
                  like settings...
                * ``ui.TYPEWINDOW`` (1): The ui-file must be a `QMainWindow` or its
                  outer widget is turned into a main window. This window is then shown.
                  This is recommended for \"standalone\" windows, that should be able
                  to be minimized, maximized, contain menus or toolbars etc.
                * ``ui.TYPEDOCKWIDGET`` (2): The loaded widget is the content of a dock
                  widget (toolbox) and is added to the indicated ``dockWidgetArea``
                  of the main window of `itom`.
                * ``ui.TYPECENTRALWIDGET`` (3): The loaded ui-file must define a
                  `QWidget` or `QMainWindow` and is then added to the central area of
                  `itom`, above the command line. It is not allowed to choose this type
                  if the user interface is created from a `QDialog`.
            dialogButtonBar (int): This argument is only used if \
                ``type == ui.TYPEDIALOG`` and defines if a button bar with buttons,
                given by ``dialogButtons`` should be automatically added to the dialog.
                If this is the case, the role of the buttons is considered, such that
                clicking the ``OK`` or ``Cancel`` button  will automatically close the
                dialog and return the role to the :meth:`show` method (if the dialog
                is displayed modal). Allowed values:

                * ``ui.BUTTONBAR_NO`` (0): do not add any button bar and buttons
                  (default),
                * ``ui.BUTTONBAR_HORIZONTAL`` (1): add a horizontal button bar at the
                  bottom,
                * ``ui.BUTTONBAR_VERTICAL`` (2): add vertical button bar on the right
                  side.

            dialogButtons (dict): Only relevant if ``dialogButtonBar`` is not
                ``ui.BUTTONBAR_NO``: This dictionary contains all buttons, that should
                be added to the button bar. For every entry, the key is the role name
                of the button (enum ``QDialogButtonBox::ButtonRole``, e.g. 'AcceptRole',
                'RejectRole', 'ApplyRole', 'YesRole', 'NoRole'). The value is
                the text of the button.
            childOfMainWindow (bool): For type ``ui.TYPEDIALOG`` and ``ui.TYPEWINDOW``
                only: Indicates if the window should be a child of the itom main
                window. If ``False``, this window has its own icon in the taskbar
                of the operating system.
            deleteOnClose (bool): Indicates if the widget / window / dialog should
                be deleted if the user closes it or if it is hidden. If it is hidden,
                it can be shown again using :meth:`show`.
            dockWidgetArea (int): Only for ``type == ui.TYPEDOCKWIDGET (2)``. Indicates
                the position where the dock widget should be placed:

                * 1 : ``ui.LEFTDOCKWIDGETAREA``
                * 2 : ``ui.RIGHTDOCKWIDGETAREA``
                * 4 : ``ui.TOPDOCKWIDGETAREA``
                * 8 : ``ui.BOTTOMDOCKWIDGETAREA``
            **kwds (keyword-based parameters): further parameters, that are passed
                to the super constructor of this. This is given, if your class derives
                from ``ItomUi`` and another class. Then, the ``kwds`` parameters
                are passed to the other class.
        """
        dialogButtons = dialogButtons or {}
        self.gui = ui(
            filename,
            type,
            dialogButtonBar,
            dialogButtons,
            childOfMainWindow,
            deleteOnClose,
            dockWidgetArea,
        )
        # this is to have s cooperative multi-inheritance structure enabled.
        # check www.realpython.com/python-super
        # check www.code.activestate.com/recipes/(577720-how-to-use-super-effectively
        # check www.rhettinger.wordpress.com/2011/05/26/super-considered-super/
        super().__init__(**kwds)
        self.autoconnect()

    def show(self, modal: int = 0) -> Union[None, int]:
        """Show the gui in a model or non-modal way.

        Args:
            modal (int): 0 if gui should be shown in non-modal way (default),
                         1 if the gui should be shown modally and this method
                         returns once the gui has been closed with the exit
                         code of the gui (1: accepted, 0: rejected), or
                         2 if it should be shown modally, but this method should
                         return immediately.

        Returns:
            None if ``modal`` is 0 or 2, else ``0`` if the modally
            shown dialog has been rejected, or ``1`` if it was accepted.

        See Also:
            itom.ui.show
        """
        return self.gui.show(modal)

    def hide(self):
        """Hides the GUI.

        New in version 2.2 of this module."""
        self.gui.hide()

    def autoconnect(self):
        """Executes auto-connection between signals of widgets and methods in this class.

        Checks all methods of your class and if they have the decorator @autoslot,
        connect them with the widget's signal, if the name of the method fits to the
        requirements (see doc of autoslot-decorator)
        """
        for key in dir(self):
            value = getattr(self, key)
            if getattr(value, "hasAutoSlot", False):
                widgets = getattr(value, "widgetName", [])
                signals = getattr(value, "signature", [])
                for widget_name, signal in zip(widgets, signals):
                    try:
                        widget = eval(f"self.gui.{widget_name}")
                    except Exception:
                        if self.gui["objectName"] == widget_name:
                            widget = self.gui
                        else:
                            print(f"Auto-connection failed: Widget {widget_name} could not be found.")
                            continue
                    try:
                        widget.connect(signal, value)
                    except Exception:
                        print(f"Auto-connection failed: Widget {widget_name} has no slot {signal}.")

    def autoslot(*attr):
        """Decorator to mark methods in derived classes to be a slot for a widget signal.

        For auto-connecting your method with a signal of a widget in the
        user interface, your method must have as name 'on_WIDGETNAME_SIGNALNAME' and
        you have to decorate your method with the decorator '@autoslot('parameters').
        """

        def decorate(func):
            """Internal decorator method."""
            parts = func.__name__.split("_")
            if len(parts) >= 3 and parts[0] == "on":
                func.hasAutoSlot = True
                signature = f"{parts[-1]}({attr[0]})"
                func.signature = getattr(func, "signature", []) + [signature]
                widget_name = "_".join(parts[1:-1]) if len(parts) > 3 else parts[1]
                func.widgetName = getattr(func, "widgetName", []) + [widget_name]
            return func

        return decorate

    @contextmanager
    def disableGui(
        self,
        disableItems: List[itom.uiItem] = None,
        showItems: List[itom.uiItem] = None,
        hideItems: List[itom.uiItem] = None,
        enableItems: List[itom.uiItem] = None,
        renameItems: List[Tuple[itom.uiItem, str]] = None,
        revertToInitialStateOnExit: bool = True,
        showWaitCursor: bool = True,
    ):
        """Factory function for with statement to disable parts of the GUI.

           This function can be called in a with statement to wrap a long going
           operation. If the with block is entered, given items of the GUI are
           switched to a disable state (hidden, disabled, shown...) and if the
           block is exited, the states will be reverted to the original or
           opposite value.

           The advantage of using this context function instead of "manually"
           disable and enabling the GUI during one operation is, that this
           approach will even revert to GUI to an enabled state, if the operation
           within the ``with`` block raises an unhandled exception.

           New in version 2.2 of this module.

        Args:
            disableItems (List[itom.uiItem], optional): list of :class:`itom.uiItem`, that
                should be disabled on entering the with block and reverted
                (enabled) on exiting it. Defaults to [].
            showItems (List[itom.uiItem], optional): list of :class:`itom.uiItem`, that
                should be shown on entering the with block and reverted (hidden)
                on exiting it. Defaults to [].
            hideItems (List[itom.uiItem], optional): list of :class:`itom.uiItem`, that
                should be disabled on entering the with block and reverted
                (enabled) on exiting it. Defaults to [].
            enableItems (List[itom.uiItem], optional): list of :class:`itom.uiItem`, that
                should be enabled on entering the with block and reverted
                (disabled) on exiting it. Added to version "2.5.0". Defaults to [].
            renameItems (List[Tuple[itom.uiItem, str]], optional): list of tuple of :class:`itom.uiItem`
                and str, that should be set as the text of the item. Defaults to [].
            revertToInitialStateOnExit (bool, optional): If True (default), all items
                are always reverted on exiting the with block to the state,
                they hade before (default). Else: they are always forced to
                be reverted to the opposite of the desired state on entering
                the with block. Defaults to True.
            showWaitCursor (bool, optional): If True, the wait cursor is
                shown during the execution of the with block and reverted
                to the previous value on exit. Defaults to True.

        An exemplary call of this context function is::

            disableItems = [self.gui.myItem1, self.gui.myItem2]
            showItems = []
            hideItems = [self.gui.myItem3, ]
            enableItems = []

            with self.disableGui(
                    disableItems,
                    showItems,
                    hideItems,
                    enableItems,
                    renameItems,
                    revertToInitialStateOnExit=True,
                    showWaitCursor=True):
                doSomethingLong()
        """
        disableItems = disableItems or []
        showItems = showItems or []
        hideItems = hideItems or []
        enableItems = enableItems or []
        renameItems = renameItems or []

        try:
            if showWaitCursor:
                itom.setApplicationCursor(16)

            revertItems = []
            renamedItems = []

            for item in disableItems:
                val = item["enabled"] if revertToInitialStateOnExit else True
                if val:
                    revertItems.append((item, "enabled", val))
                item["enabled"] = False

            for item in showItems:
                val = item["visible"] if revertToInitialStateOnExit else False
                if not val:
                    revertItems.append((item, "visible", val))
                item["visible"] = True

            for item in enableItems:
                val = item["enabled"] if revertToInitialStateOnExit else False
                if not val:
                    revertItems.append((item, "enabled", val))
                item["enabled"] = True

            for item in hideItems:
                val = item["visible"] if revertToInitialStateOnExit else True
                if val:
                    revertItems.append((item, "visible", val))
                item["visible"] = False

            for item, text in renameItems:
                val = item["text"] if revertToInitialStateOnExit else text
                renamedItems.append((item, val))
                item["text"] = text

            yield

        finally:
            if showWaitCursor:
                itom.setApplicationCursor(-1)

            for item, prop, value in revertItems:
                item[prop] = value

            for item, text in renamedItems:
                item["text"] = text

    @contextmanager
    def blockSignals(self, item: Union[itom.uiItem, List[itom.uiItem]]):
        """Factory function to temporarily block signals of an widget (item).

        This function can be called in a with statement to temporarily block
        signals of the indicated widget ``item`` or the a list or tuple
        of widgets. This can for instance be used,
        if the current value or index of a spin box, slider etc. should be changed,
        without emitting the corresponding ``currentIndexChanged``... signals.

        When the with statement is entered, ``item.call('blockSignals', True)``
        is called. At exit ``item.call('blockSignals', False)`` is called again
        to revert the block. The block is also released if any exception is
        thrown within the with statement.

        New in version 2.3 of this module.

        Args:
            item (itom.uiItem or Sequence of itom.uiItem): the ``uiItem``,
                whose signals should temporarily be blocked.

        An exemplary call of this context function is::

            with self.blockSignals(self.gui.myItem1):
                self.gui.myItem1['currentIndex'] = 2
        """
        items = item if isinstance(item, (list, tuple)) else [item]

        for it in items:
            if not isinstance(it, itom.uiItem):
                raise TypeError("Item must be an itom.uiItem or a list/tuple of itom.uiItems.")

        try:
            for it in items:
                it.call("blockSignals", True)
            yield
        finally:
            for it in items:
                it.call("blockSignals", False)


# deprecated: workaround to have old version member of class UiItem again:
ItomUi.__version__ = __version__
