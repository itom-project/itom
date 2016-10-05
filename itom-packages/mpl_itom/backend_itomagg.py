"""
Render to qt from agg
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from matplotlib.externals import six

import sys
from itom import ui
from itom import uiItem
from itom import figure as itomFigure
from itom import timer as itomTimer

from matplotlib.figure import Figure

from matplotlib.backends.backend_agg import FigureCanvasAgg
from .backend_itom import FigureManagerItom
from .backend_itom import NavigationToolbar2Itom
##### Modified itom backend import
from .backend_itom import FigureCanvasItom
##### not used
from .backend_itom import show
from .backend_itom import draw_if_interactive
from .backend_itom import backend_version
######

DEBUG = False


def new_figure_manager(num, *args, **kwargs):
    """
    Create a new figure manager instance
    """
    if DEBUG: print('backend_itomagg.new_figure_manager')
    FigureClass = kwargs.pop('FigureClass', Figure)
    existingCanvas = kwargs.pop('canvas', None)
    if(existingCanvas is None):
        #make the figure 'small' such that the first resize will really increase the size
        itomFig = itomFigure(num, width = 240, height = 160)
        itomUI = itomFig.matplotlibFigure()
        embedded = False
    else:
        itomFig = None
        embedded = True
        if(isinstance(existingCanvas,uiItem)):
            itomUI = existingCanvas
        else:
            raise("keyword 'canvas' must contain an instance of uiItem")
    thisFig = FigureClass( *args, **kwargs )
    canvas = FigureCanvasItomAgg( thisFig, num, itomUI, itomFig, embedded )
    return FigureManagerItom( canvas, num, itomUI, itomFig, embedded )


class FigureCanvasItomAgg( FigureCanvasItom, FigureCanvasAgg ):
    """
    The canvas the figure renders into.  Calls the draw and print fig
    methods, creates the renderers, etc...

    Public attribute

      figure - A Figure instance
   """

    def __init__( self, figure, num, itomUI, itomFig, embeddedCanvas ):
        if DEBUG: print('FigureCanvasQtAgg: ', figure)
        FigureCanvasItom.__init__( self, figure, num, itomUI, itomFig, embeddedCanvas )
        FigureCanvasAgg.__init__( self, figure )
        self.drawRect = False
        self.canvas.call("paintRect", False, 0,0,0,0)
        self.rect = []
        self.blitbox = None
        self.canvas["enabled"]=True
        self._agg_draw_pending = None
        self.canvasInitialized = False #will be set to True once the paintEvent has been called for the first time

    def drawRectangle(self, rect):
        if DEBUG: print('FigureCanvasItomAggBase.drawRect: ', rect)
        try:
            if rect:
                self.canvas.call("paintRect", True, rect[0], rect[1], rect[2], rect[3])
            else:
                self.canvas.call("paintRect", False, 0,0,0,0)
        except RuntimeError:
            # it is possible that the figure has currently be closed by the user
            self.signalDestroyedWidget()
            print("Matplotlib figure is not available")

    def paintEvent(self):
        """
        Copy the image from the Agg canvas to the qt.drawable.
        In Qt, all drawing should be done inside of here when a widget is
        shown onscreen.
        """
        # if the canvas does not have a renderer, then give up and wait for
        # FigureCanvasAgg.draw(self) to be called
        if not hasattr(self, 'renderer'):
            return
            
        if not self.canvasInitialized:
            self.canvasInitialized = True
            #try:
            self.canvas["updatePlotOnResize"] = True
            #except Exception:
            #    pass

        if DEBUG:
            print('FigureCanvasItomAgg.paintEvent: ', self,
                  self.get_width_height())

        if self.blitbox is None:
            # matplotlib is in rgba byte order.  QImage wants to put the bytes
            # into argb format and is in a 4 byte unsigned int.  Little endian
            # system is LSB first and expects the bytes in reverse order
            # (bgra).
            stringBuffer = self.renderer._renderer.tostring_bgra()

            X = 0
            Y = 0
            W = int(self.renderer.width)
            H = int(self.renderer.height)
            try:
                self.canvas.call("paintResult", stringBuffer, X, Y, W, H, False)
            except RuntimeError as e:
                # it is possible that the figure has currently be closed by the user
                self.signalDestroyedWidget()
                print("Matplotlib figure is not available (err: %s)" % str(e))
        else:
            bbox = self.blitbox
            l, b, r, t = bbox.extents
            w = int(r) - int(l)
            h = int(t) - int(b)
            t = int(b) + h
            reg = self.copy_from_bbox(bbox)
            stringBuffer = reg.to_string_argb()
            X = int(l)
            Y = int(self.renderer.height-t)
            W = w
            H = h
            try:
                self.canvas.call("paintResult", stringBuffer, X, Y, W, H, True)
            except RuntimeError as e:
                # it is possible that the figure has currently be closed by the user
                self.signalDestroyedWidget()
                print("Matplotlib figure is not available (err: %s)" % str(e))
            self.blitbox = None

    def draw(self):
        """
        Draw the figure with Agg, and queue a request for a Qt draw.
        """
        # The Agg draw is done here; delaying causes problems with code that
        # uses the result of the draw() to update plot elements.
        self.__draw_idle_agg()

    def draw_idle(self):
        """
        Queue redraw of the Agg buffer and request Qt paintEvent.
        """
        # The Agg draw needs to be handled by the same thread matplotlib
        # modifies the scene graph from. Post Agg draw request to the
        # current event loop in order to ensure thread affinity and to
        # accumulate multiple draw requests from event handling.
        # TODO: queued signal connection might be safer than singleShot
        if not self._agg_draw_pending:
            self._agg_draw_pending = itomTimer(0, self.__draw_idle_agg, singleShot = True)

    def __draw_idle_agg(self, *args):
        #if self.height() < 0 or self.width() < 0:
        #    self._agg_draw_pending = None
        #    return
        try:
            if self.canvas:
                FigureCanvasAgg.draw(self)
                self.paintEvent()
        finally:
            self._agg_draw_pending = None

    def blit(self, bbox=None):
        """
        Blit the region in bbox
        """
        # If bbox is None, blit the entire canvas. Otherwise
        # blit only the area defined by the bbox.
        if bbox is None and self.figure:
            bbox = self.figure.bbox

        self.blitbox = bbox
        self.paintEvent()

    def print_figure(self, *args, **kwargs):
        FigureCanvasAgg.print_figure(self, *args, **kwargs)
        self.draw()


FigureCanvas = FigureCanvasItomAgg
FigureManager = FigureManagerItom