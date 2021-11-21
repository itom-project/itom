import itomPlotlyRenderer
import plotly.express as px

# import the base class ItomUi from the module itomUi in the itom-packages subfolder
from itomUi import ItomUi
from itom import ui


class PlotlyGuiDemo(ItomUi):
    
    def __init__(self):
        """Constructor."""
        ItomUi.__init__(self, "plotlyGuiDemo.ui", ui.TYPEWINDOW, deleteOnClose=True)
        self.plotlyPlot = self.gui.plotlyPlot

    @ItomUi.autoslot("")
    def on_btnClear_clicked(self):
        self.plotlyPlot.call("setHtml", "")

    @ItomUi.autoslot("")
    def on_btnPlot1_clicked(self):
        """From the bar plot demo of plotly.

        https://plotly.com/python/bar-charts/
        """
        long_df = px.data.medals_long()
        
        fig = px.bar(long_df, x="nation", y="count", color="medal", title="Long-Form Input")
        fig.show(plotHandle = self.plotlyPlot)

    @ItomUi.autoslot("")
    def on_btnPlot2_clicked(self):
        """From the distplot demo of plotly.

        https://plotly.com/python/distplot/
        """
        df = px.data.tips()
        fig = px.histogram(df, x="total_bill", y="tip", color="sex", marginal="rug",
                           hover_data=df.columns)
        fig.show(plotHandle = self.plotlyPlot)

    @ItomUi.autoslot("")
    def on_btnPlot3_clicked(self):
        """From the animations demo of plotly.
    
        https://plotly.com/python/animations/
        """
        import plotly.express as px
        
        df = px.data.gapminder()
        
        fig = px.bar(df, x="continent", y="pop", color="continent",
          animation_frame="year", animation_group="country", range_y=[0,4000000000])
        fig.show(plotHandle = self.plotlyPlot)


# create a first instance of AutoConnectExample and the gui
win = PlotlyGuiDemo()
win.show()
