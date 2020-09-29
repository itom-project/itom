# coding=iso-8859-15

"""Base class for custom user interfaces with auto-slot connections.

License information:

itom software
URL: http://www.uni-stuttgart.de/ito
Copyright (C) 2020, Institut fuer Technische Optik (ITO),
Universitaet Stuttgart, Germany

This file is part of itom.

itom is free software; you can redistribute it and/or modify it
under the terms of the GNU Library General Public Licence as published by
the Free Software Foundation; either version 2 of the Licence, or (at
your option) any later version.

itom is distributed in the hope that it will be useful, but
WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Library
General Public Licence for more details.

You should have received a copy of the GNU Library General Public License
along with itom. If not, see <http://www.gnu.org/licenses/>.
"""

from itom import ui
import itom
from contextlib import contextmanager

__version__ = "2.2.0"


class ItomUi:
    """Base class which can be inherited in order to show an user defined
    user-interface. This class provides possibilites for auto-connecting
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
    
    def __init__(self, filename, *args, **kwds):
        """Constructor.
        
        Call this constructor from a derived class like you would
        directly call the constructor of the class ``itom.ui``.
        
        Args:
            filename (str): the path to the user interface file (*.ui)
            args (unnamed parameters): further parameters, that are
                passed to the constructor of ``itom.ui``.
            kwds (keyword-based parameters): further parameters, that are
                passed to the constructor of ``itom.ui``.
        
        Possible named or unnamed parameters can be:
        
        * type (int): ui.TYPEDIALOG, ui.TYPEWINDOW, ui.TYPEDOCKWIDGET,
          ui.TYPECENTRALWIDGET
        * childOfMainWindow (bool): only valid for ui.TYPEDIALOG or
          ui.TYPEWINDOW. True if gui is a child of the itom main window (default),
          else False.
        * deleteOnClose (bool): If True, the GUI is destroyed if it has been
          closed by the user (you can connect to the ``destroyed`` signal.
          If False (default), the GUI is only hidden and can be shown again.
        """
        self.gui = ui(filename, *args, **kwds)
        self.autoconnect()

    def show(self, modal=0):
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
                wid = getattr(value, "widgetName", [])
                sig = getattr(value, "signature", [])
                for w, s in zip(wid, sig):
                    try:
                        widget = eval("self.gui." + w)
                    except Exception:
                        if self.gui["objectName"] == w:
                            widget = self.gui
                        else:
                            print(
                                "Auto-connection failed: "
                                "Widget %s could not be found." % str(w)
                            )
                            continue

                    try:
                        widget.connect(s, value)
                    except Exception:
                        print(
                            "Auto-connection failed. Widget %s has no slot %s(%s)"
                            % (str(w), str(s), str(sig))
                        )

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
                setattr(func, "hasAutoSlot", True)
                newSig = "{0}({1})".format(parts[len(parts) - 1], attr[0])
                sig = getattr(func, "signature", [])
                sig.append(newSig)
                wid = getattr(func, "widgetName", [])

                if len(parts) == 3:
                    widgetName = parts[1]
                else:
                    widgetName = "_".join(parts[1 : len(parts) - 1])

                wid.append(widgetName)
                setattr(func, "signature", sig)
                setattr(func, "widgetName", wid)
            return func

        return decorate

    @contextmanager
    def disableGui(
        self,
        disableItems=[],
        showItems=[],
        hideItems=[],
        enableItems=[],
        revertToInitialStateOnExit=True,
        showWaitCursor=True
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
            disableItems (List[itom.uiItem]): list of :class:`itom.uiItem`, that
                should be disabled on entering the with block and reverted 
                (enabled) on exiting it.
            showItems (List[itom.uiItem]): list of :class:`itom.uiItem`, that
                should be shown on entering the with block and reverted (hidden)
                on exiting it.
            hideItems (List[itom.uiItem]): list of :class:`itom.uiItem`, that
                should be hidden on entering the with block and reverted 
                (shown) on exiting it.
            enableItems (List[itom.uiItem]): list of :class:`itom.uiItem`, that
                should be enabled on entering the with block and reverted 
                (disabled) on exiting it.
            revertToInitialStateOnExit (bool): If True (default), all items
                are always reverted on exiting the with block to the state,
                they hade before (default). Else: they are always forced to
                be reverted to the opposite of the desired state on entering
                the with block.
            showWaitCursor (bool): If True (default), the wait cursor is
                shown during the execution of the with block and reverted
                to the previous value on exit.
        
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
                    revertToInitialStateOnExit=True,
                    showWaitCursor=True):
                doSomethingLong()
        """
        try:
            # enter block
            if showWaitCursor:
                itom.setApplicationCursor(16)
            
            revertItems = []
            
            for item in disableItems:
                if revertToInitialStateOnExit:
                    val = item["enabled"]
                else:
                    val = True
                
                if val:
                    revertItems.append([item, "enabled", val])
                
                item["enabled"] = False
            
            for item in showItems:
                if revertToInitialStateOnExit:
                    val = item["visible"]
                else:
                    val = False
                
                if not val:
                    revertItems.append([item, "visible", val])
                
                item["visible"] = True
            
            for item in enableItems:
                if revertToInitialStateOnExit:
                    val = item["enabled"]
                else:
                    val = False
                
                if not val:
                    revertItems.append([item, "enabled", val])
                
                item["enabled"] = True
            
            for item in hideItems:
                if revertToInitialStateOnExit:
                    val = item["visible"]
                else:
                    val = True
                
                if val:
                    revertItems.append([item, "visible", val])
                
                item["visible"] = False
            
            yield  # this statement starts the long operation in the with block
            
        finally:
            # exit block
            if showWaitCursor:
                itom.setApplicationCursor(-1)
            
            for item, prop, value in revertItems:
                item[prop] = value


# deprecated: workaround to have old version member of class UiItem again:
ItomUi.__version__ = __version__