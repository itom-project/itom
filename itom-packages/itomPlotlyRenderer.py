# coding=iso-8859-15

"""Creates a new renderer for the Python plotly library and registers
this renderer as another possible renderer for the plotly package under
the name ``itom``. This renderer is also set as default.

Import this module before using plotly in itom such that all plotly
figures and outputs will be plot in itom figures.

If you would like to send the plotly output to a given widget of type
``PlotlyPlot``, pass its ``plotItem`` handle to the optional keyword
argument ``plotHandle`` of ``plotly.graph_objects.Figure``.
"""

import plotly.io as pio
from plotly.io._base_renderers import ExternalRenderer
import itom


class ItomPlotlyRenderer(ExternalRenderer):
    def __init__(
        self,
        config=None,
        auto_play=False,
        post_script=None,
        animation_opts=None,
    ):

        self.config = config
        self.auto_play = auto_play
        self.post_script = post_script
        self.animation_opts = animation_opts
        self._plotHandle = None

    @property
    def plotHandle(self):
        return self._plotHandle

    @plotHandle.setter
    def plotHandle(self, value):
        self._plotHandle = value

    def render(self, fig_dict):
        from plotly.io import to_html

        html = to_html(
            fig_dict,
            config=self.config,
            auto_play=self.auto_play,
            include_plotlyjs=True,
            include_mathjax="cdn",
            post_script=self.post_script,
            full_html=True,
            animation_opts=self.animation_opts,
            default_width="100%",
            default_height="100%",
            validate=False,
        )

        if self._plotHandle:
            self._plotHandle.call("setHtml", html)
        else:
            windowUi = itom.figure()  # itom figure window
            plotlyCanvas = windowUi.plotlyFigure()
            plotlyCanvas.call("setHtml", html)


pio.renderers["itom"] = ItomPlotlyRenderer()
pio.renderers.default = "itom"
