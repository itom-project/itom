# coding=iso-8859-15
from __future__ import absolute_import, division, print_function, unicode_literals

import itom
from itom import uiItem, timer, ui
from itom import figure as itomFigure

import os
import re
import sys
import weakref

import matplotlib

from matplotlib.cbook import is_string_like
from matplotlib.backend_bases import FigureManagerBase
from matplotlib.backend_bases import FigureCanvasBase
from matplotlib.backend_bases import NavigationToolbar2

from matplotlib.backend_bases import cursors
from matplotlib.backend_bases import TimerBase
from matplotlib.backend_bases import ShowBase

from matplotlib._pylab_helpers import Gcf
from matplotlib.figure import Figure

from matplotlib.widgets import SubplotTool

figureoptions = None

backend_version = "1.0.0"

DEBUG = False

# SPECIAL_KEYS are keys that do *not* return their unicode name
# instead they have manually specified names
SPECIAL_KEYS = {
    0x01000021: "control",
    0x01000020: "shift",
    0x01000023: "alt",
    0x01000022: "super",
    0x01000005: "enter",
    0x01000012: "left",
    0x01000013: "up",
    0x01000014: "right",
    0x01000015: "down",
    0x01000000: "escape",
    0x01000030: "f1",
    0x01000031: "f2",
    0x01000032: "f3",
    0x01000033: "f4",
    0x01000034: "f5",
    0x01000035: "f6",
    0x01000036: "f7",
    0x01000037: "f8",
    0x01000038: "f9",
    0x01000039: "f10",
    0x0100003A: "f11",
    0x0100003B: "f12",
    0x01000010: "home",
    0x01000011: "end",
    0x01000016: "pageup",
    0x01000017: "pagedown",
    0x01000001: "tab",
    0x01000003: "backspace",
    0x01000005: "enter",
    0x01000006: "insert",
    0x01000007: "delete",
    0x01000008: "pause",
    0x0100000A: "sysreq",
    0x0100000B: "clear",
}

# define which modifier keys are collected on keyboard events.
# elements are (mpl names, Modifier Flag, Qt Key) tuples
SUPER = 0
ALT = 1
CTRL = 2
SHIFT = 3
MODIFIER_KEYS = [
    ("super", 0x10000000, 0x01000022),
    ("alt", 0x08000000, 0x01000023),
    ("ctrl", 0x04000000, 0x01000021),
    ("shift", 0x02000000, 0x01000020),
]

if sys.platform == "darwin":
    # in OSX, the control and super (aka cmd/apple) keys are switched, so
    # switch them back.
    SPECIAL_KEYS.update(
        {0x01000021: "super", 0x01000022: "control",}  # cmd/apple key
    )
    MODIFIER_KEYS[0] = ("super", 0x04000000, 0x01000021)
    MODIFIER_KEYS[2] = ("ctrl", 0x10000000, 0x01000022)


def fn_name():
    return sys._getframe(1).f_code.co_name


cursord = {
    -1: -1,
    cursors.MOVE: 9,
    cursors.HAND: 13,
    cursors.POINTER: 0,
    cursors.SELECT_REGION: 2,
}

if hasattr(cursors, "WAIT"):  # > matplotlib 2.1
    cursord[cursors.WAIT] = 3


def draw_if_interactive():
    """
    Is called after every pylab drawing command
    """
    if matplotlib.is_interactive():
        figManager = Gcf.get_active()
        if figManager is not None:
            figManager.canvas.draw_idle()


if matplotlib.__version__ < "2.1.0":

    class Show(ShowBase):
        def mainloop(self):
            pass


else:

    class Show(ShowBase):
        @classmethod
        def mainloop(cls):
            pass


show = Show()


def new_figure_manager(num, *args, **kwargs):
    """
    Create a new figure manager instance
    """
    # thisFig = Figure( *args, **kwargs )
    FigureClass = kwargs.pop("FigureClass", Figure)
    existingCanvas = kwargs.pop("canvas", None)
    if existingCanvas is None:
        embeddedCanvas = False
        itomFig = itomFigure(num)
        itomUI = itomFig.matplotlibFigure()
        # itomUI.show() #in order to get the right size
    else:
        embeddedCanvas = True
        itomFig = None
        if isinstance(existingCanvas, uiItem):
            itomUI = existingCanvas
        else:
            raise ("keyword 'canvas' must contain an instance of uiItem")
    thisFig = FigureClass(*args, **kwargs)
    canvas = FigureCanvasItom(thisFig, num, itomUI, itomFig, embeddedCanvas)
    manager = FigureManagerItom(canvas, num, itomUI, itomFig, embeddedCanvas)
    return manager


class TimerItom(TimerBase):
    """
    Subclass of :class:`backend_bases.TimerBase` that uses itom timer events.

    Attributes:
    * interval: The time between timer events in milliseconds. Default
        is 1000 ms.
    * single_shot: Boolean flag indicating whether this timer should
        operate as single shot (run once and then stop). Defaults to False.
    * callbacks: Stores list of (func, args) tuples that will be called
        upon timer events. This list can be manipulated directly, or the
        functions add_callback and remove_callback can be used.
    """

    def __init__(self, *args, **kwargs):
        TimerBase.__init__(self, *args, **kwargs)

        # Create a new timer and connect the timeout() signal to the
        # _on_timer method.
        self._timer = itom.timer(
            self._interval, self._on_timer, singleShot=self._single
        )
        self._timer_set_interval()

    def __del__(self):
        # Probably not necessary in practice, but is good behavior to
        # disconnect
        try:
            TimerBase.__del__(self)
            self._timer = None
        except RuntimeError:
            # Timer C++ object already deleted
            pass

    def _timer_set_single_shot(self):
        self._timer = itom.timer(
            self._interval, self._on_timer, singleShot=self._single
        )

    def _timer_set_interval(self):
        self._timer.setInterval(self._interval)

    def _timer_start(self):
        self._timer.start()

    def _timer_stop(self):
        self._timer.stop()


class FigureCanvasItom(FigureCanvasBase):

    # map Qt button codes to MouseEvent's ones:
    # left 1, middle 2, right 3
    buttond = {
        1: 1,
        2: 3,
        4: 2,
    }

    def __init__(self, figure, num, itomUI, itomFig, embeddedCanvas):
        if DEBUG:
            print("FigureCanvasItom: ", figure)

        FigureCanvasBase.__init__(self, figure=figure)
        self.num = num

        self.initialized = False
        self.embeddedCanvas = embeddedCanvas
        self._destroying = False
        self.itomFig = itomFig
        # self.showEnable = False #this will be set to True if the draw() command has been called for the first time e.g. by show() of the manager

        if embeddedCanvas == False:
            self.canvas = (
                itomUI.canvasWidget
            )  # this object is deleted in the destroy-method of manager, due to cyclic garbage collection
            win = self.canvas
            # win["width"]=w
            # win["height"]=h
            win[
                "mouseTracking"
            ] = False  # by default, the itom-widget only sends mouse-move events if at least one button is pressed or the tracker-button is is checked-state
        else:
            self.canvas = itomUI.canvasWidget
            itomUI[
                "mouseTracking"
            ] = False  # by default, the itom-widget only sends mouse-move events if at least one button is pressed or the tracker-button is is checked-state

        self.canvas.connect("eventLeaveEnter(bool)", self.leaveEnterEvent)
        self.canvas.connect("eventMouse(int,int,int,int)", self.mouseEvent)
        self.canvas.connect("eventWheel(int,int,int,int)", self.wheelEvent)
        self.canvas.connect("eventKey(int,int,int,bool)", self.keyEvent)
        self.canvas.connect("eventResize(int,int)", self.resizeEvent)
        self.canvas.connect("eventCopyToClipboard(int)", self.copyToClipboardEvent)
        if matplotlib.__version__ < "2.1.0":
            # idle_event is deprecated from 2.1.0 on (removed without replacement, since usually unused)
            self.canvas.connect("eventIdle()", self.idle_event)

        w, h = self.get_width_height()
        self.resize(w, h)
        self.initialized = True

    def destroy(self):
        if self.initialized == True:
            del self.canvas
            self.canvas = None
            del self.figure  # from base class
            self.figure = None

        self.initialized = False

    def leaveEnterEvent(self, enter):
        """itom specific:
        replacement of enterEvent and leaveEvent of Qt5 backend
        """
        if enter:
            FigureCanvasBase.enter_notify_event(self, None)
        else:
            FigureCanvasBase.leave_notify_event(self, None)

    def mouseEvent(self, type, x, y, button):
        # flipy so y=0 is bottom of canvas
        y = self.figure.bbox.height - y

        try:
            # button: left 1, middle 2, right 3
            if type == 0:  # mousePressEvent
                FigureCanvasBase.button_press_event(self, x, y, button)
                if DEBUG:
                    print("button pressed:", button)
            elif type == 1:  # mouseDoubleClickEvent
                FigureCanvasBase.button_press_event(self, x, y, button, dblclick=True)
                if DEBUG:
                    print("button doubleclicked:", button)
            elif type == 2:  # mouseMoveEvent
                if (
                    button == 0
                ):  # if move without button press, reset timer since no other visualization is given to Qt, which could then reset the timer
                    self.canvas.call("stopTimer")
                FigureCanvasBase.motion_notify_event(self, x, y)
                if DEBUG:
                    print("mouse move, (x,y):", x, y)
            elif type == 3:  # mouseReleaseEvent
                FigureCanvasBase.button_release_event(self, x, y, button)
                if DEBUG:
                    print("button released, (x,y):", x, y)
        except RuntimeError:
            self.signalDestroyedWidget()

    def wheelEvent(self, x, y, delta, orientation):
        # flipy so y=0 is bottom of canvas
        y = self.figure.bbox.height - y
        # from QWheelEvent::delta doc
        steps = delta / 120
        if orientation == 1:  # vertical
            FigureCanvasBase.scroll_event(self, x, y, steps)
            if DEBUG:
                print("scroll event : delta = %i, steps = %i " % (delta, steps))

    def keyEvent(self, type, key, modifiers, autoRepeat):
        key = self._get_key(key, modifiers, autoRepeat)
        if key is None:
            return

        if type == 0:  # keyPressEvent
            FigureCanvasBase.key_press_event(self, key)
            if DEBUG:
                print("key press", key)
        elif type == 1:  # keyReleaseEvent
            FigureCanvasBase.key_release_event(self, key)
            if DEBUG:
                print("key release", key)

    def resizeEvent(self, w, h, draw=True):
        if DEBUG:
            print("FigureCanvasQt.resizeEvent(", w, ",", h, ")")
        if not self.figure is None:
            dpival = self.figure.dpi
            winch = w / dpival
            hinch = h / dpival
            self.figure.set_size_inches(winch, hinch, forward=False)
            status = self._agg_draw_pending
            if not draw and matplotlib.__version__ >= "2.1.0":
                self._agg_draw_pending = (
                    True  # else the following resize_event will call draw_idle, too
                )
            FigureCanvasBase.resize_event(self)
            self._agg_draw_pending = status
            if draw and matplotlib.__version__ < "2.1.0":
                self.draw_idle()

    def copyToClipboardEvent(self, dpi):
        self.copyToClipboard(dpi)

    def _get_key(self, event_key, event_mods, autoRepeat):
        if autoRepeat:
            return None

        # get names of the pressed modifier keys
        # bit twiddling to pick out modifier keys from event_mods bitmask,
        # if event_key is a MODIFIER, it should not be duplicated in mods
        mods = [
            name
            for name, mod_key, qt_key in MODIFIER_KEYS
            if event_key != qt_key and (event_mods & mod_key) == mod_key
        ]
        try:
            # for certain keys (enter, left, backspace, etc) use a word for the
            # key, rather than unicode
            key = SPECIAL_KEYS[event_key]
        except KeyError:
            # unicode defines code points up to 0x0010ffff
            # QT will use Key_Codes larger than that for keyboard keys that are
            # are not unicode characters (like multimedia keys)
            # skip these
            # if you really want them, you should add them to SPECIAL_KEYS
            MAX_UNICODE = 0x10FFFF
            if event_key > MAX_UNICODE:
                return None

            key = chr(event_key)
            # qt delivers capitalized letters.  fix capitalization
            # note that capslock is ignored
            if "shift" in mods:
                mods.remove("shift")
            else:
                key = key.lower()

        mods.reverse()
        return "+".join(mods + [key])

    def new_timer(self, *args, **kwargs):
        """
        Creates a new backend-specific subclass of
        :class:`backend_bases.Timer`.  This is useful for getting
        periodic events through the backend's native event
        loop. Implemented only for backends with GUIs.

        optional arguments:

        *interval*
            Timer interval in milliseconds

        *callbacks*
            Sequence of (func, args, kwargs) where func(*args, **kwargs)
            will be executed by the timer every *interval*.

    """
        return TimerItom(*args, **kwargs)

    def flush_events(self):
        pass

    def start_event_loop(self, timeout):
        FigureCanvasBase.start_event_loop_default(self, timeout)

    start_event_loop.__doc__ = FigureCanvasBase.start_event_loop_default.__doc__

    def stop_event_loop(self):
        FigureCanvasBase.stop_event_loop_default(self)

    stop_event_loop.__doc__ = FigureCanvasBase.stop_event_loop_default.__doc__

    def signalDestroyedWidget(self):
        """
        if the figure has been closed (e.g. by the user - clicking the close button),
        this might either be registered by the destroyed-event, catched by FigureManagerItom,
        or by any method of this class which tries to access the figure (since the destroyed
        signal is delivered with a time gap). This function should be called whenever the widget
        is not accessible any more, then the manager is closed as quick as possible, such that
        a new figure can be opened, if desired.
        """

        if self._destroying == False:
            self._destroying = True
            FigureCanvasBase.close_event(self)
            try:
                Gcf.destroy(self.num)
            except AttributeError:
                pass
                # It seems that when the python session is killed,
                # Gcf can get destroyed before the Gcf.destroy
                # line is run, leading to a useless AttributeError.


class FigureManagerItom(FigureManagerBase):
    """
    Public attributes

    canvas      : The FigureCanvas instance
    num         : The Figure number
    embeddedCanvas : True if figure is embedded in another user interface (ui), False if it is standalone
    itomFig        : instance of itom.figure for the figure windows, None if embedded
    itomUI         : instance of itom.plotItem (class: 'MatplotlibPlot') of the figure widget
    toolbar        : instance of the toolbar
    """

    def __init__(self, canvas, num, itomUI, itomFig, embeddedCanvas):
        if DEBUG:
            print("FigureManagerItom.%s" % fn_name())
        FigureManagerBase.__init__(self, canvas, num)
        self.embeddedCanvas = embeddedCanvas
        self.itomFig = itomFig
        self._shown = False

        if embeddedCanvas == False:
            self.itomUI = itomUI
            itomFig["windowTitle"] = "Figure %d" % num
            itomUI["focusPolicy"] = 0x2  # QtCore.Qt.ClickFocus
            itomUI.connect("destroyed()", self._widgetclosed)
        else:
            self.itomUI = itomUI
            # winWidget["windowTitle"] = ("Figure %d" % num)
            itomUI["focusPolicy"] = 0x2  # QtCore.Qt.ClickFocus
            itomUI.connect("destroyed()", self._widgetclosed)

        # image = os.path.join( matplotlib.rcParams['datapath'],'images','matplotlib.png' )
        # self.window.setWindowIcon(QtGui.QIcon( image ))

        self.canvas._destroying = False

        # the size of the toolbar is not handled by matplotlib, therefore ignore it.
        # any resize command only addresses the size of the canvas. not more.
        self.toolbar = self._get_toolbar(self.canvas)
        # if self.toolbar is not None:
        # [tbs_width, tbs_height] = [0,0] #itomUI.toolbar["sizeHint"]
        # pass
        # else:
        # tbs_width = 0
        # tbs_height = 0

        # resize the main window so it will display the canvas with the
        # requested size:
        cs_width, cs_height = self.canvas.get_width_height()  # canvas.sizeHint()
        # sbs_width, sbs_height = 0,0 #self.window.statusBar().sizeHint()
        self.resize(cs_width, cs_height)  # +tbs_height+sbs_height)

        if matplotlib.is_interactive():
            self.show()
            # self.window.show()

        # attach a show method to the figure for pylab ease of use
        # self.canvas.figure.show = lambda *args: self.window.show()
        self.canvas.figure.show = lambda *args: self.show()

        def notify_axes_change(fig):
            # This will be called whenever the current axes is changed
            if self.toolbar is not None:
                self.toolbar.update()

        self.canvas.figure.add_axobserver(notify_axes_change)

    def _show_message(self, s):
        raise RuntimeError("not yet implemented: _show_message")

    def _widgetclosed(self):
        if DEBUG:
            print("_widgetclosed called")
        if self.canvas._destroying:
            return
        self.canvas._destroying = True
        self.canvas.close_event()
        try:
            Gcf.destroy(self.num)
        except AttributeError:
            pass
            # It seems that when the python session is killed,
            # Gcf can get destroyed before the Gcf.destroy
            # line is run, leading to a useless AttributeError.

    def _get_toolbar(self, figureCanvas):
        # must be inited after the window, drawingArea and figure
        # attrs are set
        if matplotlib.rcParams["toolbar"] == "toolbar2":
            toolbar = NavigationToolbar2Itom(
                figureCanvas, self.itomUI, self.embeddedCanvas, True
            )
        else:
            toolbar = None
        return toolbar

    def resize(self, width, height):
        "set the canvas size in pixels"
        if "do_not_resize_window" in self.canvas.__dict__:
            # savefig or copyToClipboard will force to resize the window if dpi is higher than default (only in matplotlib >= 2.0). This is not wanted.
            if self.canvas.do_not_resize_window:
                return
        self.canvas.canvas.call("externalResize", width, height)
        self.canvas.resizeEvent(width, height, draw=self._shown)

    def show(self):
        if self.embeddedCanvas == False:
            try:
                self.itomFig.show()
            except RuntimeError:
                self._widgetclosed()
            except:
                pass
        self.canvas.draw_idle()
        self._shown = True

    def destroy(self, *args):
        if DEBUG:
            print("destroy figure manager (1)")

        # check for qApp first, as PySide deletes it in its atexit handler
        # if self.canvas._destroying: return
        if self.canvas._destroying == False:
            if self.embeddedCanvas == False:
                try:
                    self.itomUI.disconnect("destroyed()", self._widgetclosed)
                except:
                    pass
                try:
                    self.itomUI.hide()
                except:
                    pass
                try:
                    self.itomFig.hide()
                    itom.close(self.itomFig.handle)
                except:
                    pass
            else:
                try:
                    self.itomUI.disconnect("destroyed()", self._widgetclosed)
                except:
                    pass
        del self.itomUI
        self.itomUI = None
        del self.itomFig
        self.itomFig = None
        if self.toolbar:
            self.toolbar.destroy()
        if DEBUG:
            print("destroy figure manager (2)")
        self.canvas.destroy()
        self.canvas._destroying = True

    def get_window_title(self):
        try:
            return str(self.windowTitle)
        except Exception:
            return ""

    def set_window_title(self, title):
        self.windowTitle = title
        if self.embeddedCanvas == False:
            self.itomFig["windowTitle"] = "%s (Figure %d)" % (title, self.num)


class NavigationToolbar2Itom(NavigationToolbar2):
    def __init__(self, figureCanvas, itomUI, embeddedCanvas, coordinates=True):
        """ coordinates: should we show the coordinates on the right? """

        self.embeddedCanvas = embeddedCanvas
        self.itomUI = weakref.ref(itomUI)
        self.locLabel = None

        self.coordinates = coordinates
        self._idle = True
        self.subplotConfigDialog = None

        self.defaultSaveFileName = None

        NavigationToolbar2.__init__(self, figureCanvas)

    def _init_toolbar(self):
        self.basedir = os.path.join(matplotlib.rcParams["datapath"], "images")

        r = self.itomUI()
        if not r is None:
            r.actionHome.connect("triggered()", self.home)
            r.actionBack.connect("triggered()", self.back)
            r.actionForward.connect("triggered()", self.forward)
            r.actionPan.connect("triggered()", self.pan)
            r.actionZoomToRect.connect("triggered()", self.zoom)
            r.actionSubplotConfig.connect("triggered()", self.configure_subplots)
            r.actionSave.connect("triggered()", self.save_figure)

        if figureoptions is not None:
            a = self.addAction(
                self._icon("qt4_editor_options.png"), "Customize", self.edit_parameters
            )
            a.setToolTip("Edit curves line and axes parameters")

        self.buttons = {}

        # reference holder for subplots_adjust window
        self.adj_window = None

    if figureoptions is not None:

        def edit_parameters(self):
            allaxes = self.canvas.figure.get_axes()
            if len(allaxes) == 1:
                axes = allaxes[0]
            else:
                titles = []
                for axes in allaxes:
                    title = axes.get_title()
                    ylabel = axes.get_ylabel()
                    if title:
                        fmt = "%(title)s"
                        if ylabel:
                            fmt += ": %(ylabel)s"
                        fmt += " (%(axes_repr)s)"
                    elif ylabel:
                        fmt = "%(axes_repr)s (%(ylabel)s)"
                    else:
                        fmt = "%(axes_repr)s"
                    titles.append(
                        fmt % dict(title=title, ylabel=ylabel, axes_repr=repr(axes))
                    )
                item, ok = QtGui.QInputDialog.getItem(
                    self, "Customize", "Select axes:", titles, 0, False
                )
                if ok:
                    axes = allaxes[titles.index(str(item))]
                else:
                    return

            figureoptions.figure_edit(axes, self)

    def _update_buttons_checked(self):
        # sync button checkstates to match active mode
        self.itomUI().actionPan["checked"] = self._active == "PAN"
        self.itomUI().actionZoomToRect["checked"] = self._active == "ZOOM"
        if self._active == "PAN":
            self.set_cursor(cursors.MOVE)
        elif self._active == "ZOOM":
            self.set_cursor(cursors.SELECT_REGION)
        else:
            self.set_cursor(-1)

    def pan(self, *args):
        super(NavigationToolbar2Itom, self).pan(*args)
        self._update_buttons_checked()

    def zoom(self, *args):
        super(NavigationToolbar2Itom, self).zoom(*args)
        self._update_buttons_checked()

    def dynamic_update(self):
        self.canvas.draw_idle()

    def set_message(self, text):
        if self.coordinates:
            r = self.itomUI()
            if not r is None:
                s2 = text.encode("latin-1", "backslashreplace").decode("latin-1")
                r.call("setLabelText", s2.replace(", ", "\n"))

    def set_cursor(self, cursor):
        self.canvas.canvas.call("setCursors", cursord[cursor])

    def draw_rubberband(self, event, x0, y0, x1, y1):
        if DEBUG:
            print("draw_rubberband: ", event, x0, y0, x1, y1)

        height = self.canvas.figure.bbox.height
        y1 = height - y1
        y0 = height - y0

        w = abs(x1 - x0)
        h = abs(y1 - y0)

        rect = [round(val) for val in (min(x0, x1), min(y0, y1), w, h)]
        self.canvas.drawRectangle(rect)

    def remove_rubberband(self):
        self.canvas.drawRectangle(None)

    def configure_subplots(self):
        if self.subplotConfigDialog is None:
            self.subplotConfigDialog = SubplotToolItom(
                self.canvas.figure, self.itomUI(), self.embeddedCanvas
            )

        self.subplotConfigDialog.showDialog()

    def save_figure(self, *args):
        filetypes = self.canvas.get_supported_filetypes_grouped()
        sorted_filetypes = list(filetypes.items())
        sorted_filetypes.sort()

        if not self.defaultSaveFileName is None:
            start = self.defaultSaveFileName
            default_filetype = os.path.splitext(start)[1]
            if default_filetype == "":
                default_filetype = self.canvas.get_default_filetype()
            elif default_filetype.startswith("."):
                default_filetype = default_filetype[1:]
        else:
            default_filetype = self.canvas.get_default_filetype()
            start = "image." + default_filetype

        filters = []
        selectedFilterIndex = 0
        for name, exts in sorted_filetypes:
            exts_list = " ".join(["*.%s" % ext for ext in exts])
            filter = "%s (%s)" % (name, exts_list)
            if default_filetype in exts:
                selectedFilterIndex = len(filters)
            filters.append(filter)
        filters = ";;".join(filters)

        fname = ui.getSaveFileName(
            "Choose a filename to save to",
            start,
            filters,
            selectedFilterIndex,
            parent=self.itomUI(),
        )

        if fname:
            try:
                self.canvas.print_figure(str(fname))
                self.defaultSaveFileName = fname
            except Exception as e:
                ui.msgCritical("Error saving file", str(e), parent=self.itomUI())

    def set_history_buttons(self):
        if hasattr(self, "_views"):  # matplotlib <= 2.1.0
            can_backward = self._views._pos > 0
            can_forward = self._views._pos < len(self._views._elements) - 1
        elif hasattr(self, "_nav_stack"):  # matplotlib
            can_backward = self._nav_stack._pos > 0
            can_forward = self._nav_stack._pos < len(self._nav_stack._elements) - 1

        itomUI = self.itomUI()
        if not itomUI is None:
            itomUI.actionBack["enabled"] = can_backward
            itomUI.actionForward["enabled"] = can_forward

    def destroy(self):
        del self.canvas  # in base class
        self.canvas = None


class SubplotToolItom(SubplotTool):
    def __init__(self, targetfig, itomUI, embeddedCanvas):
        # SubplotTool.__init__(self, targetfig, targetfig)

        self.targetfig = targetfig
        self.embeddedCanvas = embeddedCanvas
        self.itomUI = weakref.ref(itomUI)
        itomUI.connect("subplotConfigSliderChanged(int,int)", self.funcgeneral)
        itomUI.connect("subplotConfigTight()", self.functight)
        itomUI.connect("subplotConfigReset()", self.reset)

    def _setSliderPositions(self):
        valLeft = int(self.targetfig.subplotpars.left * 1000)
        valBottom = int(self.targetfig.subplotpars.bottom * 1000)
        valRight = int(self.targetfig.subplotpars.right * 1000)
        valTop = int(self.targetfig.subplotpars.top * 1000)
        valWSpace = int(self.targetfig.subplotpars.wspace * 1000)
        valHSpace = int(self.targetfig.subplotpars.hspace * 1000)

        r = self.itomUI()
        if not r is None:
            r.call(
                "modifySubplotSliders",
                valLeft,
                valTop,
                valRight,
                valBottom,
                valWSpace,
                valHSpace,
            )

    def showDialog(self):
        self.defaults = {}
        for attr in (
            "left",
            "bottom",
            "right",
            "top",
            "wspace",
            "hspace",
        ):
            val = getattr(self.targetfig.subplotpars, attr)
            self.defaults[attr] = val
        self._setSliderPositions()

        valLeft = int(self.targetfig.subplotpars.left * 1000)
        valBottom = int(self.targetfig.subplotpars.bottom * 1000)
        valRight = int(self.targetfig.subplotpars.right * 1000)
        valTop = int(self.targetfig.subplotpars.top * 1000)
        valWSpace = int(self.targetfig.subplotpars.wspace * 1000)
        valHSpace = int(self.targetfig.subplotpars.hspace * 1000)

        r = self.itomUI()
        if not r is None:
            r.call(
                "showSubplotConfig",
                valLeft,
                valTop,
                valRight,
                valBottom,
                valWSpace,
                valHSpace,
            )

    def funcgeneral(self, type, val):
        if type == 0:
            self.funcleft(val)
        elif type == 1:
            self.functop(val)
        elif type == 2:
            self.funcright(val)
        elif type == 3:
            self.funcbottom(val)
        elif type == 4:
            self.funcwspace(val)
        elif type == 5:
            self.funchspace(val)

    def funcleft(self, val):
        # if val == self.sliderright.value():
        #    val -= 1
        self.targetfig.subplots_adjust(left=val / 1000.0)
        if self.drawon:
            self.targetfig.canvas.draw_idle()

    def funcright(self, val):
        # if val == self.sliderleft.value():
        #    val += 1
        self.targetfig.subplots_adjust(right=val / 1000.0)
        if self.drawon:
            self.targetfig.canvas.draw_idle()

    def funcbottom(self, val):
        # if val == self.slidertop.value():
        #    val -= 1
        self.targetfig.subplots_adjust(bottom=val / 1000.0)
        if self.drawon:
            self.targetfig.canvas.draw_idle()

    def functop(self, val):
        # if val == self.sliderbottom.value():
        #    val += 1
        self.targetfig.subplots_adjust(top=val / 1000.0)
        if self.drawon:
            self.targetfig.canvas.draw_idle()

    def funcwspace(self, val):
        self.targetfig.subplots_adjust(wspace=val / 1000.0)
        if self.drawon:
            self.targetfig.canvas.draw_idle()

    def funchspace(self, val):
        self.targetfig.subplots_adjust(hspace=val / 1000.0)
        if self.drawon:
            self.targetfig.canvas.draw_idle()

    def functight(self):
        self.targetfig.tight_layout()
        self._setSliderPositions()
        self.targetfig.canvas.draw_idle()

    def reset(self):
        self.targetfig.subplots_adjust(**self.defaults)
        self._setSliderPositions()
        self.targetfig.canvas.draw_idle()


FigureCanvas = FigureCanvasItom
FigureManager = FigureManagerItom
