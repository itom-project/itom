"""
Render to qt/itom from agg
"""


import os, sys

import matplotlib
from matplotlib.figure import Figure

from matplotlib.backends.backend_agg import FigureCanvasAgg
from .backend_itom import FigureManagerItom, FigureCanvasItom,\
     show, draw_if_interactive, \
     NavigationToolbar2Itom
     
from itom import uiItem, ui

DEBUG = False


def new_figure_manager( num, *args, **kwargs ):
    """
    Create a new figure manager instance
    """
    if DEBUG: print('backend_qtagg.new_figure_manager')
    FigureClass = kwargs.pop('FigureClass', Figure)
    existingCanvas = kwargs.pop('canvas', None)
    if(existingCanvas is None):
        itomUI = ui("itom://matplotlib")
        embedded = False
    else:
        embedded = True
        if(isinstance(existingCanvas,uiItem)):
            itomUI = existingCanvas
        else:
            raise("keyword 'canvas' must contain an instance of uiItem")
    thisFig = FigureClass( *args, **kwargs )
    canvas = FigureCanvasItomAgg( thisFig, num, itomUI, embedded )
    return FigureManagerItom( canvas, num, itomUI, embedded )

class NavigationToolbar2ItomAgg(NavigationToolbar2Itom):
    def _get_canvas(self, fig):
        return FigureCanvasItomAgg(fig)

class FigureManagerItomAgg(FigureManagerItom):
    def _get_toolbar(self, canvas, parent):
        # must be inited after the window, drawingArea and figure
        # attrs are set
        if matplotlib.rcParams['toolbar']=='classic':
            print("Classic toolbar is not supported")
        elif matplotlib.rcParams['toolbar']=='toolbar2':
            toolbar = NavigationToolbar2ItomAgg(canvas, parent)
        else:
            toolbar = None
        return toolbar

class FigureCanvasItomAgg( FigureCanvasItom, FigureCanvasAgg ):
    """
    The canvas the figure renders into.  Calls the draw and print fig
    methods, creates the renderers, etc...

    Public attribute

      figure - A Figure instance
   """

    def __init__( self, figure, num, itomUI, embeddedCanvas ):
        if DEBUG: print('FigureCanvasQtAgg: ', figure)
        FigureCanvasItom.__init__( self, figure, num, itomUI, embeddedCanvas )
        FigureCanvasAgg.__init__( self, figure )
        self.drawRect = False
        self.canvas.call("paintRect", False, 0,0,0,0)
        self.rect = []
        self.blitbox = None
        #self.setAttribute(QtCore.Qt.WA_OpaquePaintEvent)
        self.canvas.connect("eventPaintRequest()", self.paintEvent)
        self.canvas["enabled"]=True #only if enabled is true, the widget starts to trigger the eventPaintRequest-signal

    def drawRectangle( self, rect ):
        if DEBUG: print('FigureCanvasItomAgg.drawRect: ', rect)
        self.rect = rect
        self.drawRect = True
        
        try:
            self.canvas.call("paintRect", True, rect[0], rect[1], rect[2], rect[3])
        except RuntimeError:
            # it is possible that the figure has currently be closed by the user
            self.signalDestroyedWidget()
            print("Matplotlib figure is not available")

    def paintEvent( self):
        """
        Copy the image from the Agg canvas to the qt.drawable.
        In Qt, all drawing should be done inside of here when a widget is
        shown onscreen.
        """

        #FigureCanvasItom.paintEvent( self, e )
        if DEBUG: print('FigureCanvasItomAgg.paintEvent: ', self, \
           self.get_width_height())
        
        if self.blitbox is None:
            # matplotlib is in rgba byte order.  QImage wants to put the bytes
            # into argb format and is in a 4 byte unsigned int.  Little endian
            # system is LSB first and expects the bytes in reverse order
            # (bgra).
            
            #if QtCore.QSysInfo.ByteOrder == QtCore.QSysInfo.LittleEndian:
            stringBuffer = self.renderer._renderer.tostring_bgra()
            #else:
            #    stringBuffer = self.renderer._renderer.tostring_argb()
            #
            
            XYrect = 0
            WHrect = 0
            
            #if self.drawRect:
            #    XYrect = (int(self.rect[0]) << 16) + int(self.rect[1])
            #    WHrect = (int(self.rect[2]) << 16) + int(self.rect[3])
            X = 0
            Y = 0
            W = int(self.renderer.width)
            H = int(self.renderer.height)
            try:
                self.canvas.call("paintResult", stringBuffer, X, Y, W, H, False)
            except RuntimeError:
                # it is possible that the figure has currently be closed by the user
                self.signalDestroyedWidget()
                print("Matplotlib figure is not available")
        else:
            bbox = self.blitbox
            l, b, r, t = bbox.extents
            w = int(r) - int(l)
            h = int(t) - int(b)
            t = int(b) + h
            reg = self.copy_from_bbox(bbox)
            stringBuffer = reg.to_string_argb()
            #stringBuffer = reg.to_string_bgra()
            
            X = int(l)
            Y = int(self.renderer.height-t)
            W = w
            H = h
            XY = (int(l) << 16) + int(self.renderer.height-t)
            WH = (w << 16) + h
            try:
                self.canvas.call("paintResult", stringBuffer, X, Y, W, H, True)
            except RuntimeError:
                # it is possible that the figure has currently be closed by the user
                self.signalDestroyedWidget()
                print("Matplotlib figure is not available")
            self.blitbox = None
        self.drawRect = False
    

        
    
    def draw( self ):
        """
        Draw the figure with Agg, and queue a request
        for a Qt draw.
        """
        # The Agg draw is done here; delaying it until the paintEvent
        # causes problems with code that uses the result of the
        # draw() to update plot elements.
        if DEBUG: print('FigureCanvasItomAgg.draw')
        FigureCanvasAgg.draw(self)
        self.paintEvent()
        #self.canvas.call("update")

    def blit(self, bbox=None):
        """
        Blit the region in bbox
        """
        self.blitbox = bbox
        l, b, w, h = bbox.bounds
        t = b + h
        #self.repaint(l, self.renderer.height-t, w, h)
        self.paintEvent()

    def print_figure(self, *args, **kwargs):
        FigureCanvasAgg.print_figure(self, *args, **kwargs)
        self.draw()
