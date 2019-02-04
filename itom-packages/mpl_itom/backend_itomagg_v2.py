"""
Render to qt from agg
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from matplotlib import cbook
from matplotlib.transforms import Bbox

from matplotlib.backends.backend_agg import FigureCanvasAgg
from .backend_itom_v2 import FigureCanvasItom, _BackendItom


class FigureCanvasItomAgg( FigureCanvasItom, FigureCanvasAgg ):

    def __init__(self, figure, num, matplotlibWidgetUiItem, embeddedWidget):
        FigureCanvasItom.__init__( self, figure, num, matplotlibWidgetUiItem, embeddedWidget)
        FigureCanvasAgg.__init__( self, figure )
        self._bbox_queue = []
        self.canvasInitialized = False #will be set to True once the paintEvent has been called for the first time

    @property
    @cbook.deprecated("2.1")
    def blitbox(self):
        return self._bbox_queue

    def paintEvent(self):
        """Copy the image from the Agg canvas to the itom plugin 'matplotlibWidgetUiItem'.

        In itom, all drawing should be done inside of here when a widget is
        shown onscreen.
        """
        if not self.canvasInitialized:
            self.canvasInitialized = True
            self.matplotlibWidgetUiItem["updatePlotOnResize"] = True
            
        if self._update_dpi():
            # The dpi update triggered its own paintEvent.
            return
        self.draw_idle()  # Only does something if a draw is pending.

        # if the canvas does not have a renderer, then give up and wait for
        # FigureCanvasAgg.draw(self) to be called
        if not hasattr(self, 'renderer'):
            return

        #painter = QtGui.QPainter(self) (not for itom)

        if self._bbox_queue:
            bbox_queue = self._bbox_queue
        else:
            #painter.eraseRect(self.rect()) (not for itom)
            bbox_queue = [
                Bbox([[0, 0], [self.renderer.width, self.renderer.height]])]
        self._bbox_queue = []
        for bbox in bbox_queue:
            l, b, r, t = map(int, bbox.extents)
            w = r - l
            h = t - b
            
            #<-- itom specific start
            reg = self.copy_from_bbox(bbox)
            buf = reg.to_string_argb()
            '''qimage = QtGui.QImage(buf, w, h, QtGui.QImage.Format_ARGB32)
            # Adjust the buf reference count to work around a memory leak bug
            # in QImage under PySide on Python 3.
            if QT_API == 'PySide' and six.PY3:
                ctypes.c_long.from_address(id(buf)).value = 1
            if hasattr(qimage, 'setDevicePixelRatio'):
                # Not available on Qt4 or some older Qt5.
                qimage.setDevicePixelRatio(self._dpi_ratio)
            origin = QtCore.QPoint(l, self.renderer.height - t)
            #painter.drawImage(origin / self._dpi_ratio, qimage)'''
            W = round(w)
            H = round(h)
            #workaround sometimes the width and hight does not fit to the buf length, leding to a crash of itom.
            #If the length is a multiple of either the width or the length we readjust them.
            if not int(W*H*4) ==len(buf):
                numberElements= len(buf)/4
                if numberElements%H == 0:
                    W=int(numberElements/H)
                elif numberElements%W == 0:
                    H=int(numberElements/W)
                else:
                    return
            try:
                self.matplotlibWidgetUiItem.call("paintResult", buf, 0, 0, W, H, False)
            except RuntimeError as e:
                # it is possible that the figure has currently be closed by the user
                self.signalDestroyedWidget()
                print("Matplotlib figure is not available (err: %s)" % str(e))
            #itom specific end -->
            
    def copyToClipboard(self, dpi):
        if not hasattr(self, 'renderer'):
            return
        
        origDPI = self.figure.dpi
        origfacecolor = self.figure.get_facecolor()
        origedgecolor = self.figure.get_edgecolor()
        
        #if dpi is higher than the current value, matplotlib >= 2.0 will temporarily set the new dpi and
        #force the window to be resized. We do not want a resized window, but only an internal change
        #of the dpi value. The property 'do_not_resize_window' is considered in FigureManagerItom.resize
        #and the resize is not executed if do_not_resize_window is True.
        self.do_not_resize_window = True

        self.figure.dpi = dpi
        self.figure.set_facecolor('w')
        self.figure.set_edgecolor('w')
        
        FigureCanvasAgg.draw(self)
        
        renderer = self.get_renderer()
        original_dpi = renderer.dpi
        renderer.dpi = dpi
        stringBuffer = renderer._renderer.tostring_bgra()
        width = renderer.width
        height = renderer.height
        renderer.dpi = original_dpi
        try:
            self.matplotlibWidgetUiItem.call("copyToClipboardResult", stringBuffer, 0, 0, width, height)
        except RuntimeError as e:
            # it is possible that the figure has currently be closed by the user
            self.signalDestroyedWidget()
            print("Matplotlib figure is not available (err: %s)" % str(e))
            
        self.figure.dpi = origDPI
        self.figure.set_facecolor(origfacecolor)
        self.figure.set_edgecolor(origedgecolor)
        
        self.do_not_resize_window = False
        
        self.figure.set_canvas(self)

    def blit(self, bbox=None):
        """Blit the region in bbox.
        """
        # If bbox is None, blit the entire canvas. Otherwise
        # blit only the area defined by the bbox.
        if bbox is None and self.figure:
            bbox = self.figure.bbox

        self._bbox_queue.append(bbox)

        # repaint uses logical pixels, not physical pixels like the renderer.
        l, b, w, h = [pt / self._dpi_ratio for pt in bbox.bounds]
        t = b + h
        self.paintEvent()
        #self.repaint(l, self.renderer.height / self._dpi_ratio - t, w, h)

    def print_figure(self, *args, **kwargs):
        #<-- itom specific start
        #if dpi is higher than the current value, matplotlib >= 2.0 will temporarily set the new dpi and
        #force the window to be resized. We do not want a resized window, but only an internal change
        #of the dpi value. The property 'do_not_resize_window' is considered in FigureManagerItom.resize
        #and the resize is not executed if do_not_resize_window is True.
        self.do_not_resize_window = True
        #itom specific end -->
        
        super(FigureCanvasItomAgg, self).print_figure(*args, **kwargs)
        
        #<-- itom specific start
        self.do_not_resize_window = False
        #itom specific end -->
        
        self.draw()


#@cbook.deprecated("2.2")
#class FigureCanvasQTAggBase(FigureCanvasQTAgg):
#    pass


@_BackendItom.export
class _BackendItomAgg(_BackendItom):
    FigureCanvas = FigureCanvasItomAgg
