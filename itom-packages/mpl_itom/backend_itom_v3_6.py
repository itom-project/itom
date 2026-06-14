# coding=iso-8859-15
import functools
import os
import re
import signal
import sys
import traceback

import matplotlib

from matplotlib import backend_tools, cbook
from matplotlib._pylab_helpers import Gcf
from matplotlib.backend_bases import (
    _Backend,
    FigureCanvasBase,
    FigureManagerBase,
    NavigationToolbar2,
    TimerBase,
    cursors,
    ToolContainerBase,
    _Mode,
    MouseButton,
    CloseEvent,
    KeyEvent,
    LocationEvent,
    MouseEvent,
    ResizeEvent
)

import mpl_itom.figureoptions as figureoptions

# from matplotlib.backends.qt_editor.formsubplottool import UiSubplotTool
from matplotlib.figure import Figure
from matplotlib.backend_managers import ToolManager

# itom specific imports
import itom
from itom import uiItem, timer, ui
import weakref

# itom specific imports (end)

backend_version = "3.2.1"
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
        {0x01000021: "cmd", 0x01000022: "control"}  # cmd/apple key
    )
    MODIFIER_KEYS[0] = ("cmd", 0x04000000, 0x01000021)
    MODIFIER_KEYS[2] = ("ctrl", 0x10000000, 0x01000022)


cursord = {
    -1: -1,  # --> itom specific: remove current overwrite cursor
    cursors.MOVE: 9,
    cursors.HAND: 13,
    cursors.POINTER: 0,
    cursors.SELECT_REGION: 2,
    cursors.WAIT: 3,
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


class TimerItom(TimerBase):
    """
    Subclass of :class:`backend_bases.TimerBase` that uses Qt timer events.

    Attributes
    ----------
    interval : int
        The time between timer events in milliseconds. Default is 1000 ms.
    single_shot : bool
        Boolean flag indicating whether this timer should
        operate as single shot (run once and then stop). Defaults to False.
    callbacks : list
        Stores list of (func, args) tuples that will be called upon timer
        events. This list can be manipulated directly, or the functions
        `add_callback` and `remove_callback` can be used.

    """

    def __init__(self, *args, **kwargs):
        # Create a new timer and connect the timeout() signal to the
        # _on_timer method.
        # set a long default interval to stop the timer. The super
        # constructor will then directly set the interval and singleShot.
        self._timer = itom.timer(
            1000000, self._on_timer, singleShot=False
        )
        super().__init__(*args, **kwargs)

    def __del__(self):
        # Probably not necessary in practice, but is good behavior to
        # disconnect
        try:
            self._timer_stop()
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

    # map itom/matplotlibWidget button codes to MPL button codes
    # left 1, middle 2, right 3, no mouse button 0
    # todo: from MPL 3.1 on, these values can directly be mapped
    # to MouseButton.LEFT, MouseButton.RIGHT, MouseButton.MIDDLE...
    buttond = {
        0: 0,
        1: MouseButton.LEFT,
        2: MouseButton.MIDDLE,
        3: MouseButton.RIGHT,
        # QtCore.Qt.XButton1: None,
        # QtCore.Qt.XButton2: None,
    }

    def __init__(self, figure, num, matplotlibplotUiItem, embeddedWidget):
        """
        embeddedWidget: bool
            True, if matplotlib widget is embedded into a loaded ui file,
            False: matplotlib widget is displayed in figure window
        """
        super().__init__(figure=figure)

        self.figure = figure

        # --> itom specific start
        self.num = num

        self.initialized = False
        self.embeddedWidget = embeddedWidget
        self.itomUI = matplotlibplotUiItem  # for historic reasons only
        self._destroying = False
        # self.showEnable = False #this will be set to True if the draw() command has been called for the first time e.g. by show() of the manager

        self.matplotlibWidgetUiItem = (
            matplotlibplotUiItem.canvasWidget
        )  # this object is deleted in the destroy-method of manager, due to cyclic garbage collection
        self.matplotlibWidgetUiItem[
            "mouseTracking"
        ] = True  # by default, the itom-widget only sends mouse-move events if at least one button is pressed or the tracker-button is is checked-state

        self.matplotlibWidgetUiItem.connect("eventEnter(int,int)", self.enterEvent)
        self.matplotlibWidgetUiItem.connect("eventLeave()", self.leaveEvent)
        self.matplotlibWidgetUiItem.connect(
            "eventMouse(int,int,int,int)", self.mouseEvent
        )
        self.matplotlibWidgetUiItem.connect(
            "eventWheel(int,int,int,int)", self.wheelEvent
        )
        self.matplotlibWidgetUiItem.connect("eventKey(int,int,int,bool)", self.keyEvent)
        self.matplotlibWidgetUiItem.connect("eventResize(int,int)", self.resizeEvent)
        self.matplotlibWidgetUiItem.connect(
            "eventCopyToClipboard(int)", self.copyToClipboardEvent
        )
        # itom specific end <--

        # We don't want to scale up the figure DPI more than once.
        # Note, we don't handle a signal for changing DPI yet.
        figure._original_dpi = figure.dpi
        self._update_figure_dpi()
        # In cases with mixed resolution displays, we need to be careful if the
        # dpi_ratio changes - in this case we need to resize the canvas
        # accordingly. We could watch for screenChanged events from Qt, but
        # the issue is that we can't guarantee this will be emitted *before*
        # the first paintEvent for the canvas, so instead we keep track of the
        # dpi_ratio value here and in paintEvent we resize the canvas if
        # needed.
        self._dpi_ratio_prev = None

        self._draw_pending = False
        self._erase_before_paint = False
        self._is_drawing = False

        # self.setAttribute(QtCore.Qt.WA_OpaquePaintEvent)
        # self.setMouseTracking(True)
        # resize is done by manager constructor later
        # self.resize(*self.get_width_height())
        # Key auto-repeat enabled by default
        self._keyautorepeat = True

        # --> itom specific start
        self.initialized = True
        # itom specific end <--

        self.lastResizeSize = (0, 0)

    def destroy(self):
        """itom specific function. not in qt5 backend"""
        if self.initialized == True:
            del self.matplotlibWidgetUiItem
            self.matplotlibWidgetUiItem = None
            del self.figure  # from base class
            self.figure = None

        self.initialized = False

    def _update_figure_dpi(self):
        dpi = self._dpi_ratio * self.figure._original_dpi
        self.figure._set_dpi(dpi, forward=False)

    @property
    def _dpi_ratio(self):
        """
        itom: currently only returns 1
        todo: implement the following code from the qt5 backend:
        # Not available on Qt4 or some older Qt5.
        try:
            # self.devicePixelRatio() returns 0 in rare cases
            return self.devicePixelRatio() or 1
        except AttributeError:
            return 1
        """
        try:
            dpi_ratio = self.matplotlibWidgetUiItem.call("devicePixelRatioF")
        except Exception:
            dpi_ratio = 1
        return dpi_ratio

    def _update_dpi(self):
        # As described in __init__ above, we need to be careful in cases with
        # mixed resolution displays if dpi_ratio is changing between painting
        # events.
        # Return whether we triggered a resizeEvent (and thus a paintEvent)
        # from within this function.
        if self._dpi_ratio != self._dpi_ratio_prev:
            if DEBUG:
                print("update dpi ratio to %.2f" % self._dpi_ratio)
            # We need to update the figure DPI.
            self._update_figure_dpi()
            self._dpi_ratio_prev = self._dpi_ratio
            # The easiest way to resize the canvas is to emit a resizeEvent
            # since we implement all the logic for resizing the canvas for
            # that event.

            # --> itom specific start
            width, height = self.matplotlibWidgetUiItem["size"]
            self.matplotlibWidgetUiItem.call("externalResize", width, height)
            # itom specific end <--

            ##--> begin itom specific code
            # externalResize does not trigger a paintEvent. therefore return False
            return False
            ## end itom specific code <--
            """# resizeEvent triggers a paintEvent itself, so we exit this one
            # (after making sure that the event is immediately handled).
            return True"""
        return False

    def get_width_height(self):
        w, h = FigureCanvasBase.get_width_height(self)
        if self.matplotlibWidgetUiItem.exists():
            return int(w / self._dpi_ratio), int(h / self._dpi_ratio)
        else:
            return 0, 0

    def set_cursor(self, cursor):
        # docstring inherited
        self.matplotlibWidgetUiItem.call("setCursor", cursord[cursor])

    def enterEvent(self, x, y):
        """itom specific:
        replacement of enterEvent and leaveEvent of Qt5 backend
        """
        x_, y_ = self.mouseEventCoords(x, y)
        LocationEvent("figure_enter_event", self,
                      x_, y_,
                      guiEvent=None)._process()

    def leaveEvent(self):
        """itom specific:
        replacement of enterEvent and leaveEvent of Qt5 backend
        """
        itom.setApplicationCursor(-1)
        LocationEvent("figure_leave_event", self,
                      0, 0,
                      guiEvent=None)._process()

    def mouseEventCoords(self, x, y):
        """Calculate mouse coordinates in physical pixels

        Qt5 use logical pixels, but the figure is scaled to physical
        pixels for rendering.   Transform to physical pixels so that
        all of the down-stream transforms work as expected.

        Also, the origin is different and needs to be corrected.

        """
        # flip y so y=0 is bottom of canvas
        if self.figure is not None:
            y = self.figure.bbox.height - y
        else:
            # figure has already been closed.
            y, x = 0, 0

        return x, y

    def mouseEvent(self, eventType, x, y, button):
        x, y = self.mouseEventCoords(x, y)
        button = self.buttond.get(button)
        if button is None:
            button = 0  # fallback solution
        if DEBUG:
            print("mouseEvent %s (%.2f,%.2f), button: %s" % (eventType, x, y, button))
        try:
            # button: left 1, middle 2, right 3
            if eventType == 0:  # mousePressEvent
                MouseEvent("button_press_event",
                           self, x, y, button,
                           guiEvent=None)._process()
            elif eventType == 1:  # mouseDoubleClickEvent
                MouseEvent("button_press_event",
                           self, x, y, button, dblclick=True,
                           guiEvent=None)._process()
            elif eventType == 2:  # mouseMoveEvent
                if button == 0:
                    # if move without button press, reset timer since no other
                    # visualization is given to Qt, which could then reset the timer
                    self.matplotlibWidgetUiItem.call("stopTimer")
                MouseEvent("motion_notify_event", self,
                   x, y,
                   guiEvent=None)._process()
            elif eventType == 3:  # mouseReleaseEvent
                MouseEvent("button_release_event", self,
                       x, y, button,
                       guiEvent=None)._process()
        except NotImplementedError:
            # derived from RuntimeError, therefore handle it separately.
            pass
        except RuntimeError:
            self.signalDestroyedWidget()

    def wheelEvent(self, x, y, delta, orientation):
        x, y = self.mouseEventCoords(x, y)
        # from QWheelEvent::delta doc
        steps = delta / 120
        if orientation == 1:  # vertical
            MouseEvent("scroll_event", self,
                       x, y, step=steps,
                       guiEvent=None)._process()

    def keyEvent(self, type, key, modifiers, autoRepeat):
        key = self._get_key(key, modifiers, autoRepeat)
        if key is None:
            return

        if type == 0:  # keyPressEvent
            # mouse coordinates are missing here
            KeyEvent("key_press_event", self,
                     key, 0, 0,
                     guiEvent=None)._process()
        elif type == 1:  # keyReleaseEvent
            KeyEvent("key_release_event", self,
                     key, 0, 0,
                     guiEvent=None)._process()

    def resizeEvent(self, w, h, draw=True):
        if self._destroying or (w, h) == self.lastResizeSize:
            return

        if DEBUG:
            print(
                "resizeEvent: %i, %i, %i, last: %s"
                % (w, h, draw, str(self.lastResizeSize))
            )

        # _dpi_ratio_prev will be set the first time the canvas is painted, and
        # the rendered buffer is useless before anyways.
        if self._dpi_ratio_prev is None:
            return
        if not self.figure is None and self.matplotlibWidgetUiItem.exists():
            dpival = self.figure.dpi
            winch = self._dpi_ratio * w / dpival
            hinch = self._dpi_ratio * h / dpival
            self.figure.set_size_inches(winch, hinch, forward=False)
            status = self._is_drawing
            if not draw:
                self._is_drawing = (
                    True  # else the following resize_event will call draw_idle, too
                )
            # emit our resize events
            ResizeEvent("resize_event", self)._process()
            self._is_drawing = status
            self.lastResizeSize = (w, h)
            self.draw_idle()

    def copyToClipboardEvent(self, dpi):
        self.copyToClipboard(dpi)

    def _get_key(self, event_key, event_mods, autoRepeat):
        if not self._keyautorepeat and autoRepeat:
            return None

        ##event_key = event.key()
        ##event_mods = int(event.modifiers())  # actually a bitmask

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

        Other Parameters
        ----------------
        interval : scalar
            Timer interval in milliseconds

        callbacks : list
            Sequence of (func, args, kwargs) where ``func(*args, **kwargs)``
            will be executed by the timer every *interval*.

        """
        return TimerItom(*args, **kwargs)

    def flush_events(self):
        itom.processEvents()

    def start_event_loop(self, timeout=0):
        raise NotImplementedError("itom backend does not support interactive mode")
        ##if hasattr(self, "_event_loop") and self._event_loop.isRunning():
        ##    raise RuntimeError("Event loop already running")
        ##self._event_loop = event_loop = QtCore.QEventLoop()
        ##if timeout:
        ##    timer = QtCore.QTimer.singleShot(timeout * 1000, event_loop.quit)
        ##event_loop.exec_()

    def stop_event_loop(self, event=None):
        raise NotImplementedError("itom backend does not support interactive mode")
        ##if hasattr(self, "_event_loop"):
        ##    self._event_loop.quit()

    def draw(self):
        """Render the figure, and queue a request for a Qt draw.
        """
        # The renderer draw is done here; delaying causes problems with code
        # that uses the result of the draw() to update plot elements.
        if self._is_drawing:
            return
        with cbook._setattr_cm(self, _is_drawing=True):
            super().draw()
        self._erase_before_paint = True
        self.paintEvent()

    def draw_idle(self):
        """Queue redraw of the Agg buffer and request Qt paintEvent.
        """
        # The Agg draw needs to be handled by the same thread matplotlib
        # modifies the scene graph from. Post Agg draw request to the
        # current event loop in order to ensure thread affinity and to
        # accumulate multiple draw requests from event handling.
        # TODO: queued signal connection might be safer than singleShot
        if not (self._draw_pending or self._is_drawing):
            self._draw_pending = True
            self._draw_idle()


    def _draw_idle(self):
        # if self.height() < 0 or self.width() < 0:
        #    self._draw_pending = False
        if not self._draw_pending:
            return
        try:
            if self.matplotlibWidgetUiItem:
                self.draw()
        except Exception:
            # Uncaught exceptions are fatal for PyQt5, so catch them instead.
            traceback.print_exc()
        finally:
            self._draw_pending = False

    def drawRectangle(self, rect):
        # Draw the zoom rectangle to the QPainter.  _draw_rect_callback needs
        # to be called at the end of paintEvent.
        try:
            if rect:
                self.matplotlibWidgetUiItem.call(
                    "paintRect", True, *(pt / (1+0*self._dpi_ratio) for pt in rect)
                )
            else:
                self.matplotlibWidgetUiItem.call("paintRect", False, 0, 0, 0, 0)
        except RuntimeError:
            # it is possible that the figure has currently be closed by the user
            self.signalDestroyedWidget()
            print("Matplotlib figure is not available")

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
            CloseEvent("close_event", self)._process()
            try:
                Gcf.destroy(self.num)
            except AttributeError:
                pass
                # It seems that when the python session is killed,
                # Gcf can get destroyed before the Gcf.destroy
                # line is run, leading to a useless AttributeError.


class FigureManagerItom(FigureManagerBase):
    """
    Attributes
    ----------
    canvas : `FigureCanvas`
        The FigureCanvas instance
    num : int or str
        The Figure number
    toolbar : qt.QToolBar
        The qt.QToolBar
    windowUi : Optional[itom.figure]
        None or itom.Figure of outer window

    """

    def __init__(self, canvas, num, matplotlibplotUiItem, windowUi, embeddedWidget):

        self.canvas = canvas
        self.windowUi = windowUi  # can also be None if embeddedWidget is True
        self.matplotlibplotUiItem = matplotlibplotUiItem
        self.matplotlibWidgetUiItem = self.matplotlibplotUiItem
        self.itomUI = self.matplotlibplotUiItem  # for historic reasons only
        self.embeddedWidget = embeddedWidget
        self._shown = False

        self.matplotlibplotUiItem["focusPolicy"] = 0x2  # QtCore.Qt.ClickFocus
        self.matplotlibplotUiItem.connect("destroyed()", self._widgetclosed)

        super().__init__(canvas, num)

        if embeddedWidget == False and self.windowUi:
            self.windowUi["windowTitle"] = "Figure %d" % num

        """self.windowUi.closing.connect(canvas.close_event)
        self.windowUi.closing.connect(self._widgetclosed)

        self.windowUi.setWindowTitle("Figure %d" % num)
        image = os.path.join(matplotlib.rcParams['datapath'],
                             'images', 'matplotlib.svg')
        self.windowUi.setWindowIcon(QtGui.QIcon(image))

        # Give the keyboard focus to the figure instead of the
        # manager; StrongFocus accepts both tab and click to focus and
        # will enable the canvas to process event w/o clicking.
        # ClickFocus only takes the focus is the window has been
        # clicked
        # on. http://qt-project.org/doc/qt-4.8/qt.html#FocusPolicy-enum or
        # http://doc.qt.digia.com/qt/qt.html#FocusPolicy-enum
        self.canvas.setFocusPolicy(QtCore.Qt.StrongFocus)
        self.canvas.setFocus()"""

        self.canvas._destroying = False

        self.toolmanager = self._get_toolmanager()
        self.toolbar = self._get_toolbar(self.canvas, self.windowUi)

        if self.toolmanager:
            backend_tools.add_tools_to_manager(self.toolmanager)
            if self.toolbar:
                backend_tools.add_tools_to_container(self.toolbar)

        if self.toolbar is not None:
            # self.windowUi.addToolBar(self.toolbar)
            if not self.toolmanager:
                # add text label to status bar (make it self, since the connect command only holds a weakref)
                self.statusbar_label = self.matplotlibplotUiItem.call("statusBar").call(
                    "addLabelWidget", "statusbarLabel"
                )
                self.toolbar.message.connect(self.statusbar_label, property="text")
            # tbs_height = self.toolbar.sizeHint().height()
        else:
            # tbs_height = 0
            pass

        # resize the main window so it will display the canvas with the
        # requested size:
        cs_width, cs_height = self.canvas.get_width_height()
        self.resize(cs_width, cs_height)
        """cs = canvas.sizeHint()
        sbs = self.windowUi.statusBar().sizeHint()
        self._status_and_tool_height = tbs_height + sbs.height()
        height = cs.height() + self._status_and_tool_height
        self.windowUi.resize(cs.width(), height)

        self.windowUi.setCentralWidget(self.canvas)"""

        if matplotlib.is_interactive():
            self.windowUi.show()
            self.canvas.draw_idle()

    def full_screen_toggle(self):
        raise NotImplementedError("no fullscreen mode available for itom backend")
        ##if self.windowUi.isFullScreen():
        ##    self.windowUi.showNormal()
        ##else:
        ##    self.windowUi.showFullScreen()

    def _widgetclosed(self):
        if self.canvas._destroying:
            return
        self.canvas._destroying = True
        if matplotlib.__version__ >= "3.8.0":
            self.canvas.flush_events()
        else:
            self.canvas.close_event()
        try:
            Gcf.destroy(self.num)
        except AttributeError:
            pass
            # It seems that when the python session is killed,
            # Gcf can get destroyed before the Gcf.destroy
            # line is run, leading to a useless AttributeError.

    def _get_toolbar(self, canvas, parent):
        # must be inited after the window, drawingArea and figure
        # attrs are set
        if matplotlib.rcParams["toolbar"] == "toolbar2":
            matplotlibplotUiItem = self.matplotlibplotUiItem
            toolbar = NavigationToolbar2Itom(
                canvas, matplotlibplotUiItem, parent, False
            )
        elif matplotlib.rcParams["toolbar"] == "toolmanager":
            toolbar = ToolbarItom(self.toolmanager, self.matplotlibplotUiItem)
        else:
            toolbar = None
        return toolbar

    def _get_toolmanager(self):
        if matplotlib.rcParams["toolbar"] == "toolmanager":
            toolmanager = ToolManager(self.canvas.figure)
        else:
            toolmanager = None
        return toolmanager

    def resize(self, width, height):
        """set the canvas size in pixels"""
        if "do_not_resize_window" in self.canvas.__dict__:
            # savefig or copyToClipboard will force to resize the window if dpi
            # is higher than default (only in matplotlib >= 2.0). This is not wanted.
            if self.canvas.do_not_resize_window:
                return
        self.matplotlibWidgetUiItem.canvasWidget.call("externalResize", width, height)
        self.canvas.resizeEvent(width, height, draw=self._shown)

    def show(self):
        if self.embeddedWidget == False:
            try:
                self.windowUi.show()
            except RuntimeError:
                self._widgetclosed()
            except:
                pass
        self.canvas.draw_idle()
        self._shown = True

    def destroy(self, *args):
        # check for qApp first, as PySide deletes it in its atexit handler
        # if self.canvas._destroying: return
        if self.canvas._destroying == False:
            if self.embeddedWidget == False:
                try:
                    self.matplotlibplotUiItem.disconnect(
                        "destroyed()", self._widgetclosed
                    )
                except:
                    pass
                try:
                    self.matplotlibplotUiItem.hide()
                except:
                    pass
                try:
                    self.windowUi.hide()
                    itom.close(self.windowUi.handle)
                except:
                    pass
            else:
                try:
                    self.matplotlibplotUiItem.disconnect(
                        "destroyed()", self._widgetclosed
                    )
                except:
                    pass
        del self.matplotlibplotUiItem
        self.matplotlibplotUiItem = None
        del self.windowUi
        self.windowUi = None
        if self.toolbar:
            self.toolbar.destroy()
        self.canvas.destroy()
        self.canvas._destroying = True

    def get_window_title(self):
        try:
            return str(self.windowTitle)
        except Exception:
            return ""

    def set_window_title(self, title):
        self.windowTitle = title
        if self.embeddedWidget == False:
            self.windowUi["windowTitle"] = "%s (Figure %d)" % (title, self.num)


class Signal:
    def __init__(self):
        self.callbacks = []

    def emit(self, *args):
        for c in self.callbacks:
            widget = c["uiItem"]()
            if widget:
                if c["property"]:
                    widget[c["property"]] = args[0]
                elif c["slot"]:
                    widget.call(c["slot"], *args)

    def connect(self, widget, property=None, slot=None):
        if property is not None:
            self.callbacks.append({"uiItem": weakref.ref(widget), "property": property})
        else:
            self.callbacks.append({"uiItem": weakref.ref(widget), "slot": slot})


class NavigationToolbar2Itom(NavigationToolbar2):
    def __init__(self, canvas, matplotlibplotUiItem, parentUi, coordinates=True):
        """ coordinates: should we show the coordinates on the right? """
        self.canvas = canvas
        self.parentUi = parentUi
        self.matplotlibplotUiItem = weakref.ref(matplotlibplotUiItem)
        self.coordinates = coordinates
        # self._actions = {}
        self.message = Signal()
        self.subplotConfigDialog = None

        """initializes the toolbar."""
        self._initToolbar()

        NavigationToolbar2.__init__(self, canvas)

    def _get_predef_action(self, callback_name):
        w = self.matplotlibplotUiItem()
        if callback_name == "home":
            return w.actionHome
        elif callback_name == "back":
            return w.actionBack
        elif callback_name == "forward":
            return w.actionForward
        elif callback_name == "pan":
            return w.actionPan
        elif callback_name == "zoom":
            return w.actionZoomToRect
        elif callback_name == "configure_subplots":
            return w.actionSubplotConfig
        elif callback_name == "save_figure":
            return w.actionSave
        elif callback_name == "copy_clipboard":
            return w.actionCopyClipboard
        return None

    def _icon_filename(self, name):
        if name.startswith(":"):
            return name
        return name.replace(".png", "_large.png")

    def _action_name(self, name):
        objectName = "action_%s" % name
        return re.sub(
            "[^a-zA-Z0-9_]", "_", objectName
        )  # replace all characters, which are not among the given set, by an underscore

    def _init_toolbar(self):
        """Empty method for backward compatibility.
        This method must be kept empty from MPL 3.3.0 on.
        Its content is now contained in the method _initToolbar.
        """

    def _initToolbar(self):
        """
        """
        w = self.matplotlibplotUiItem()
        if not w:
            return

        items = self.toolitems
        callbacks = [i[3] for i in items]
        if "configure_subplots" in callbacks and not "subplots" in callbacks:
            icon = (
                ":/itomDesignerPlugins/general/icons/settings"  # self._icon_filename()
            )
            items = items + (
                (
                    "Edit parameters...",
                    "Edit axis, curve and image parameters",
                    icon,
                    "edit_parameters",
                ),
            )

        for text, tooltip_text, image_file, callback in items:
            if text is None:
                # add separator
                self.matplotlibplotUiItem().call(
                    "addUserDefinedAction", "", "", "", "", "default"
                )
            else:
                a = self._get_predef_action(callback)
                if not a:
                    action_name = self._action_name(callback)
                    self.matplotlibplotUiItem().call(
                        "addUserDefinedAction",
                        action_name,
                        text,
                        self._icon_filename(image_file + ".png"),
                        "",
                        "default",
                    )

                    a = eval("self.matplotlibplotUiItem().%s" % action_name)
                else:
                    a["visible"] = True

                if callback in ["zoom", "pan"]:
                    a["checkable"] = True
                elif callback == "save_figure":
                    a2 = self._get_predef_action("copy_clipboard")
                    if a2:
                        a2["visible"] = True

                if a["checkable"]:
                    a.connect("triggered(bool)", getattr(self, callback))
                else:
                    a.connect("triggered()", getattr(self, callback))

                if tooltip_text is not None:
                    a["toolTip"] = tooltip_text

        # not used in MPL >= 3.3.0 any more
        self.buttons = {}

        # reference holder for subplots_adjust window
        # not used in MPL >= 3.3.0 any more
        self.adj_window = None

    def edit_parameters(self):
        allaxes = self.canvas.figure.get_axes()
        if not allaxes:
            itom.ui.msgWarning(
                "Error",
                "There are no axes to edit.",
                parent=self.matplotlibplotUiItem(),
            )
            return
        elif len(allaxes) == 1:
            (axes,) = allaxes
        else:
            titles = []
            for axes in allaxes:
                name = (
                    axes.get_title()
                    or " - ".join(filter(None, [axes.get_xlabel(), axes.get_ylabel()]))
                    or "<anonymous {} (id: {:#x})>".format(
                        type(axes).__name__, id(axes)
                    )
                )
                titles.append(name)
            item, ok = itom.ui.getItem(
                "Customize",
                "Select axes:",
                titles,
                0,
                False,
                parent=self.matplotlibplotUiItem(),
            )
            if ok:
                axes = allaxes[titles.index(item)]
            else:
                return

        figureoptions.figure_edit(self.matplotlibplotUiItem(), axes, self)

    def _update_buttons_checked(self):
        # sync button checkstates to match active mode
        self._get_predef_action("pan")["checked"] = self.mode == _Mode.PAN
        self._get_predef_action("zoom")["checked"] = self.mode == _Mode.ZOOM

    def pan(self, *args):
        super().pan(*args)
        self._update_buttons_checked()

    def zoom(self, *args):
        super().zoom(*args)
        self._update_buttons_checked()

    def set_message(self, s):
        # text, that is shown in the statusbar should not contain multi
        # line texts. Therefore replace it by ', '.
        s_ = s.replace("\n", ", ")
        self.message.emit(s_)
        if self.coordinates:
            r = self.matplotlibplotUiItem()
            if not r is None:
                s2 = s.encode("latin-1", "backslashreplace").decode("latin-1")
                r.call("setLabelText", s2.replace(", ", "\n"))

    def set_cursor(self, cursor):
        r = self.matplotlibplotUiItem()
        if not r is None:
            r.canvasWidget.call("setCursors", cursord[cursor])

    def draw_rubberband(self, event, x0, y0, x1, y1):
        height = self.canvas.figure.bbox.height
        y1 = height - y1
        y0 = height - y0
        rect = [int(val) for val in (x0, y0, x1 - x0, y1 - y0)]
        self.canvas.drawRectangle(rect)

    def remove_rubberband(self):
        self.canvas.drawRectangle(None)

    def configure_subplots(self):
        # itom specific start
        if self.subplotConfigDialog is None:
            self.subplotConfigDialog = SubplotToolItom(
                self.canvas.figure, self.matplotlibplotUiItem()
            )
        self.subplotConfigDialog.showDialog()
        # itom specific end

    def save_figure(self, *args):
        filetypes = self.canvas.get_supported_filetypes_grouped()
        sorted_filetypes = sorted(filetypes.items())
        default_filetype = self.canvas.get_default_filetype()

        startpath = os.path.expanduser(matplotlib.rcParams["savefig.directory"])
        start = os.path.join(startpath, self.canvas.get_default_filename())
        filters = []
        selectedFilter = 0
        for name, exts in sorted_filetypes:
            exts_list = " ".join(["*.%s" % ext for ext in exts])
            filter = "%s (%s)" % (name, exts_list)
            if default_filetype in exts:
                selectedFilter = len(filters)
            filters.append(filter)
        filters = ";;".join(filters)

        fname = ui.getSaveFileName(
            "Choose a filename to save to",
            start,
            filters,
            selectedFilter,
            parent=self.parentUi,
        )
        if fname:
            # Save dir for next time, unless empty str (i.e., use cwd).
            if startpath != "":
                matplotlib.rcParams["savefig.directory"] = os.path.dirname(fname)
            try:
                self.canvas.figure.savefig(fname)
            except Exception as e:
                itom.ui.msgCritical(
                    "Error saving file",
                    str(e),
                    ui.MsgBoxOk,
                    ui.MsgBoxNoButton,
                    parent=self.parentUi,
                )

    def destroy(self):
        pass


class SubplotToolItom:
    def __init__(self, targetfig, itomUI):
        self._figure = targetfig
        self.itomUI = weakref.ref(itomUI)
        itomUI.connect("subplotConfigSliderChanged(int,int)", self.funcgeneral)
        itomUI.connect("subplotConfigTight()", self.functight)
        itomUI.connect("subplotConfigReset()", self.reset)

        self._attrs = ["top", "bottom", "left", "right", "hspace", "wspace"]
        self._defaults = {
            attr: vars(self._figure.subplotpars)[attr] for attr in self._attrs
        }

    def _setSliderPositions(self):
        valLeft = int(self._figure.subplotpars.left * 1000)
        valBottom = int(self._figure.subplotpars.bottom * 1000)
        valRight = int(self._figure.subplotpars.right * 1000)
        valTop = int(self._figure.subplotpars.top * 1000)
        valWSpace = int(self._figure.subplotpars.wspace * 1000)
        valHSpace = int(self._figure.subplotpars.hspace * 1000)

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
            val = getattr(self._figure.subplotpars, attr)
            self.defaults[attr] = val
        self._setSliderPositions()

        valLeft = int(self._figure.subplotpars.left * 1000)
        valBottom = int(self._figure.subplotpars.bottom * 1000)
        valRight = int(self._figure.subplotpars.right * 1000)
        valTop = int(self._figure.subplotpars.top * 1000)
        valWSpace = int(self._figure.subplotpars.wspace * 1000)
        valHSpace = int(self._figure.subplotpars.hspace * 1000)

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
        self._figure.subplots_adjust(left=val / 1000.0)
        self._figure.canvas.draw_idle()

    def funcright(self, val):
        self._figure.subplots_adjust(right=val / 1000.0)
        self._figure.canvas.draw_idle()

    def funcbottom(self, val):
        self._figure.subplots_adjust(bottom=val / 1000.0)
        self._figure.canvas.draw_idle()

    def functop(self, val):
        self._figure.subplots_adjust(top=val / 1000.0)
        self._figure.canvas.draw_idle()

    def funcwspace(self, val):
        self._figure.subplots_adjust(wspace=val / 1000.0)
        self._figure.canvas.draw_idle()

    def funchspace(self, val):
        self._figure.subplots_adjust(hspace=val / 1000.0)
        self._figure.canvas.draw_idle()

    def functight(self):
        self._figure.tight_layout()
        self._setSliderPositions()
        self._figure.canvas.draw_idle()

    def reset(self):
        self._figure.subplots_adjust(**self.defaults)
        self._setSliderPositions()
        self._figure.canvas.draw_idle()


class ToolbarItom(ToolContainerBase):
    def __init__(self, toolmanager, matplotlibplotUiItem):
        ToolContainerBase.__init__(self, toolmanager)
        self._toolitems = {}
        self.matplotlibplotUiItem = weakref.ref(matplotlibplotUiItem)

        # add text label to status bar (make it self, since the connect command only holds a weakref)
        self.statusbar_label = (
            self.matplotlibplotUiItem()
            .call("statusBar")
            .call("addLabelWidget", "statusbarLabel")
        )

    @property
    def _icon_extension(self):
        return "_large.png"

    def _action_name(self, name):
        objectName = "action_%s" % name
        return re.sub(
            "[^a-zA-Z0-9_]", "_", objectName
        )  # replace all characters, which are not among the given set, by an underscore

    def add_toolitem(self, name, group, position, image_file, description, toggle):

        if self.matplotlibplotUiItem() is None:
            return

        if description is None:
            description = ""

        action_name = self._action_name(name)

        parent = self.matplotlibplotUiItem()

        parent.call(
            "addUserDefinedAction",
            action_name,
            name,
            image_file,
            description,
            group,
            position,
        )

        button = eval("parent.%s" % action_name)

        def handler():
            self.trigger_tool(name)

        if toggle:
            button["checkable"] = True
        button.connect("triggered()", handler)

        self._toolitems.setdefault(name, [])
        self._toolitems[name].append((button, handler))

    def toggle_toolitem(self, name, toggled):
        if name not in self._toolitems:
            return
        for button, handler in self._toolitems[name]:
            try:
                button.disconnect("triggered()", handler)
            except Exception:
                pass
            button["checked"] = toggled
            button.connect("triggered()", handler)

    def remove_toolitem(self, name):
        if self.matplotlibplotUiItem():
            self.matplotlibplotUiItem().call(
                "removeUserDefinedAction", self._action_name(name)
            )
        del self._toolitems[name]

    def set_message(self, s):
        """
        Display a message on the toolbar (here: statusbar).

        Parameters
        ----------
        s : str
            Message text.
        """
        self.statusbar_label["text"] = s


@backend_tools._register_tool_class(FigureCanvasItom)
class ConfigureSubplotsItom(backend_tools.ConfigureSubplotsBase):
    def __init__(self, name, *args, **kwargs):
        super(backend_tools.ConfigureSubplotsBase, self).__init__(name, *args, **kwargs)
        self.subplotConfigDialog = None

    def trigger(self, *args):
        if self.subplotConfigDialog is None:
            self.subplotConfigDialog = SubplotToolItom(
                self.canvas.figure, self.canvas.manager.matplotlibplotUiItem
            )
        self.subplotConfigDialog.showDialog()


@backend_tools._register_tool_class(FigureCanvasItom)
class SaveFigureItom(backend_tools.SaveFigureBase):
    def __init__(self, *args, **kwargs):
        backend_tools.SaveFigureBase.__init__(self, *args, **kwargs)
        self.defaultSaveFileName = None

    def trigger(self, *args):
        filetypes = self.canvas.get_supported_filetypes_grouped()
        sorted_filetypes = sorted(filetypes.items())
        default_filetype = self.canvas.get_default_filetype()

        startpath = os.path.expanduser(matplotlib.rcParams["savefig.directory"])
        start = os.path.join(startpath, self.canvas.get_default_filename())
        filters = []
        selectedFilter = None
        for name, exts in sorted_filetypes:
            exts_list = " ".join(["*.%s" % ext for ext in exts])
            filtername = "%s (%s)" % (name, exts_list)
            if default_filetype in exts:
                selectedFilter = filtername
            filters.append(filtername)
        selectedFilterIndex = -1
        if selectedFilter and selectedFilter in filters:
            selectedFilterIndex = filters.index(selectedFilter)
        filters = ";;".join(filters)

        parent = self.canvas.matplotlibWidgetUiItem
        fname = ui.getSaveFileName(
            "Choose a filename to save to",
            start,
            filters,
            selectedFilterIndex,
            parent=parent,
        )

        if fname:
            # Save dir for next time, unless empty str (i.e., use cwd).
            if startpath != "":
                matplotlib.rcParams["savefig.directory"] = os.path.dirname(fname)
            try:
                self.canvas.figure.savefig(fname)
                self.defaultSaveFileName = fname
            except Exception as e:
                itom.ui.msgCritical("Error saving file", str(e), parent=parent)

if matplotlib.__version__ < "3.7.0":
    @backend_tools._register_tool_class(FigureCanvasItom)
    class SetCursorItom(backend_tools.SetCursorBase):
        def set_cursor(self, cursor):
            self.canvas.matplotlibWidgetUiItem.call("setCursor", cursord[cursor])
else:
    @backend_tools._register_tool_class(FigureCanvasItom)
    class SetCursorItom(backend_tools.ToolSetCursor):
        def set_cursor(self, cursor):
            self.canvas.matplotlibWidgetUiItem.call("setCursor", cursord[cursor])


@backend_tools._register_tool_class(FigureCanvasItom)
class RubberbandItom(backend_tools.RubberbandBase):
    def draw_rubberband(self, x0, y0, x1, y1):
        height = self.canvas.figure.bbox.height
        y1 = height - y1
        y0 = height - y0
        rect = [int(val) for val in (x0, y0, x1 - x0, y1 - y0)]
        self.canvas.drawRectangle(rect)

    def remove_rubberband(self):
        self.canvas.drawRectangle(None)


@backend_tools._register_tool_class(FigureCanvasItom)
class HelpItom(backend_tools.ToolHelpBase):
    def trigger(self, *args):
        ui.msgInformation(
            "Help", self._get_help_html(), parent=self.canvas.matplotlibWidgetUiItem
        )


@backend_tools._register_tool_class(FigureCanvasItom)
class ToolCopyToClipboardItom(backend_tools.ToolCopyToClipboardBase):
    def trigger(self, *args, **kwargs):
        self.canvas.copyToClipboard(self.canvas._dpi_ratio)


# must be None, not NavigationToolbar2Itom!
FigureManagerItom._toolbar2_class = None
# must be None, not ToolbarItom!
FigureManagerItom._toolmanager_toolbar_class = None


@_Backend.export
class _BackendItom(_Backend):
    FigureCanvas = FigureCanvasItom
    FigureManager = FigureManagerItom

    @classmethod
    def new_figure_manager(cls, num, *args, **kwargs):
        """
        Create a new figure manager instance
        """
        FigureClass = kwargs.pop("FigureClass", Figure)
        existingCanvas = kwargs.pop("canvas", None)
        if existingCanvas is None:
            embeddedWidget = False
            windowUi = itom.figure(num)  # itom figure window
            matplotlibplotUiItem = windowUi.matplotlibFigure()
        else:
            embeddedWidget = True
            windowUi = None  # matplotlib widget is embedded in a user-defined ui file
            if isinstance(existingCanvas, uiItem):
                matplotlibplotUiItem = existingCanvas
            else:
                raise ("keyword 'canvas' must contain an instance of uiItem")
        thisFig = FigureClass(*args, **kwargs)
        canvas = cls.FigureCanvas(thisFig, num, matplotlibplotUiItem, embeddedWidget)
        manager = cls.FigureManager(
            canvas, num, matplotlibplotUiItem, windowUi, embeddedWidget
        )
        return manager

    @classmethod
    def new_figure_manager_given_figure(cls, num, figure):
        """Create a new figure manager instance for the given figure.
        """
        canvas = cls.FigureCanvas(figure)
        manager = cls.FigureManager(canvas, num)
        return manager

    @staticmethod
    def mainloop():
        pass
        ### allow KeyboardInterrupt exceptions to close the plot window.
        ##signal.signal(signal.SIGINT, signal.SIG_DFL)
        ##qApp.exec_()
