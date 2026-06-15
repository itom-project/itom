"""
Render to itom (qt) from agg
"""
from matplotlib.transforms import Bbox

from matplotlib import cbook
from matplotlib.backends.backend_agg import FigureCanvasAgg
from .backend_itom_v3_6 import FigureCanvasItom, _BackendItom

import itom
import numpy as np  # for color channel conversion in copy to clipboard

DEBUG = False


class FigureCanvasItomAgg(FigureCanvasItom, FigureCanvasAgg):
    def __init__(self, figure, num, matplotlibWidgetUiItem, embeddedWidget):
        FigureCanvasItom.__init__(
            self, figure, num, matplotlibWidgetUiItem, embeddedWidget
        )
        FigureCanvasAgg.__init__(self, figure)
        self.canvasInitialized = False  # will be set to True once the paintEvent has been called for the first time

        self.paintEvent()  # paint initialization!

    def paintEvent(self, rect=None):
        """Copy the image from the Agg canvas to the itom plugin 'matplotlibWidgetUiItem'.

        In itom, all drawing should be done inside of here when a widget is
        shown onscreen.
        """

        if self._destroying:
            return

        self.paintEventTimer = None

        if DEBUG and (not rect is None):
            print("rect given:", rect)

        if not self.canvasInitialized:
            self.canvasInitialized = True
            self.matplotlibWidgetUiItem["updatePlotOnResize"] = True

        if self._update_dpi():
            # The dpi update triggered its own paintEvent.
            return
        # self._draw_idle()  # Only does something if a draw is pending.

        # if the canvas does not have a renderer, then give up and wait for
        # FigureCanvasAgg.draw(self) to be called
        if not hasattr(self, "renderer"):
            return

        if not rect is None:
            x0, y0, w, h = rect
            bbox = Bbox([[x0, y0], [w, h]])
            blit = True
        else:
            bbox = Bbox([[0, 0], [self.renderer.width, self.renderer.height]])
            blit = False

        x0, y0, w2, h2 = map(int, bbox.extents)
        w = w2 - x0
        h = h2 - y0

        if DEBUG:
            print("paint: %i,%i,%i,%i, blit=%s" % (x0, y0, w, h, str(blit)))

        # <-- itom specific start
        reg = self.copy_from_bbox(bbox)
        buf = (
            reg.to_string_argb()
        )  # this is faster than the Qt-original version cbook._unmultiplied_rgba8888_to_premultiplied_argb32...
        W = round(w)
        H = round(h)
        # workaround sometimes the width and hight does not fit to the buf length, leding to a crash of itom.
        # If the length is a multiple of either the width or the length we readjust them.
        if not int(W * H * 4) == len(buf):
            numberElements = len(buf) / 4
            if numberElements % H == 0:
                W = int(numberElements / H)
            elif numberElements % W == 0:
                H = int(numberElements / W)
            else:
                return
        try:
            # if blit: W and H are a sum of the real width/height and the offset x0 or y0.
            # else: W and H are the real width and height of the image
            self.matplotlibWidgetUiItem.call("paintResult", buf, x0, y0, W, H, blit)
        except RuntimeError as e:
            # it is possible that the figure has currently be closed by the user
            # do not call self.signalDestroyedWidget() among others,
            # since it might be, that the paintEvent is called from
            # a timer event, that should finish, such that a close event
            # is called afterwards.
            pass
            # self.signalDestroyedWidget()
            # print("paintEvent::Matplotlib figure is not available (err: %s)" % str(e))
        # itom specific end -->

    def copyToClipboard(self, dpi):
        if not hasattr(self, "renderer"):
            return

        origDPI = self.figure.dpi
        origfacecolor = self.figure.get_facecolor()
        origedgecolor = self.figure.get_edgecolor()

        # if dpi is higher than the current value, matplotlib >= 2.0 will temporarily set the new dpi and
        # force the window to be resized. We do not want a resized window, but only an internal change
        # of the dpi value. The property 'do_not_resize_window' is considered in FigureManagerItom.resize
        # and the resize is not executed if do_not_resize_window is True.
        self.do_not_resize_window = True

        self.figure.dpi = dpi
        self.figure.set_facecolor("w")
        self.figure.set_edgecolor("w")

        FigureCanvasAgg.draw(self)

        renderer = self.get_renderer()
        original_dpi = renderer.dpi
        renderer.dpi = dpi
        stringBuffer = renderer.buffer_rgba()
        width = int(renderer.width)
        height = int(renderer.height)
        arr = np.frombuffer(stringBuffer, "uint8").reshape([height, width, 4])
        stringBuffer = arr.take([2, 1, 0, 3], axis=2).tobytes()

        renderer.dpi = original_dpi
        try:
            self.matplotlibWidgetUiItem.call(
                "copyToClipboardResult", stringBuffer, 0, 0, width, height
            )
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
        if DEBUG:
            print("blit:", str(bbox))
        # If bbox is None, blit the entire canvas. Otherwise
        # blit only the area defined by the bbox.
        if bbox is None and self.figure:
            bbox = self.figure.bbox

        # repaint uses logical pixels, not physical pixels like the renderer.
        dpi_ratio = self._dpi_ratio
        x0, y0, w, h = [pt / dpi_ratio for pt in bbox.extents]

        self.paintEvent((x0, y0, w, h))

    def print_figure(self, *args, **kwargs):
        # <-- itom specific start
        # if dpi is higher than the current value, matplotlib >= 2.0 will temporarily set the new dpi and
        # force the window to be resized. We do not want a resized window, but only an internal change
        # of the dpi value. The property 'do_not_resize_window' is considered in FigureManagerItom.resize
        # and the resize is not executed if do_not_resize_window is True.
        self.do_not_resize_window = True
        # itom specific end -->

        super(FigureCanvasItomAgg, self).print_figure(*args, **kwargs)

        # <-- itom specific start
        self.do_not_resize_window = False
        # itom specific end -->

        self.draw()


# @cbook.deprecated("2.2")
# class FigureCanvasQTAggBase(FigureCanvasQTAgg):
#    pass


@_BackendItom.export
class _BackendItomAgg(_BackendItom):
    FigureCanvas = FigureCanvasItomAgg
