"""Matplotlib auto update
=========================

"""

import matplotlib
import matplotlib.pyplot as plt
from itomUi import ItomUi
from itom import ui

matplotlib.use("module://mpl_itom.backend_itomagg")
# sphinx_gallery_thumbnail_path = '11_demos/_static/_thumb/demoMatplotlibGUI.png'


class MatplotGuiAutoUpdate(ItomUi):
    def __init__(self):
        ItomUi.__init__(
            self,
            "matplotlibGuiAutoUpdate.ui",
            ui.TYPEWINDOW,
            childOfMainWindow=True,
            deleteOnClose=True,
        )
        # the deleteOnClose = True argument will not only hide the figure if the close-button is pressed,
        # but also destroy it. Then, the destroyed signal is emitted, caught by the slot self.on_mainWindow_destroyed.
        # This can be used, to delete certain things.

        self.gui.btnStop["enabled"] = False
        self.timerID = None
        self.counter = 1
        self.fignum = (
            max(
                [
                    0,
                ]
                + plt.get_fignums()
            )
            + 1
        )  # guarantee to get a new matplotlib figure number

        if not "__version__" in ItomUi.__dict__ or ItomUi.__version__ < "2.1":
            # in earlier versions of itom, an auto-connection to methods of the gui itself was not possible.
            # From ItomUi.__version__ >= 2.1 on, the dialog or window itself can be auto-connected by its
            # object name.
            self.gui.connect("destroyed()", self.on_mainWindow_destroyed)

    @ItomUi.autoslot("")
    def on_btnStart_clicked(self):
        if self.timerID is None:
            # start the timer that calls the method 'timeout' every 2 seconds
            self.timerID = timer(2000, self.timeout)
            self.timeout()
        self.gui.btnStart["enabled"] = False
        self.gui.btnStop["enabled"] = True

    @ItomUi.autoslot("")
    def on_btnStop_clicked(self):
        if not self.timerID is None:
            # stop and delete the timer, if started
            self.timerID.stop()
            self.timerID = None
        self.gui.btnStart["enabled"] = True
        self.gui.btnStop["enabled"] = False

    # for itom <= 2.1, this auto-slot will raise a runtime error, however it is manually connected in the constructor of this class.
    @ItomUi.autoslot("")
    def on_mainWindow_destroyed(self):
        """The windows was closed and destroyed. Stop the timer and tell matplotlib to close the figure"""
        if not self.timerID is None:
            self.timerID.stop()
            self.timerID = None
        plt.close(self.fignum)

    def timeout(self):
        print("update plot")

        canvas = self.gui.matplotlibPlot  # Reference to matplotlibPlot widget
        fig = plt.figure(num=self.fignum, canvas=canvas)

        if len(fig.axes) == 0:
            # create a new subplot in the figure
            ax = fig.add_subplot(111)
        else:
            # reuse the existing first subplot
            ax = fig.axes[0]
            ax.clear()
        
        ax.imshow(dataObject.randN([100, 100], "uint8"), cmap=plt.cm.gray)
        ax.set_title("title of plot [%i]" % self.counter)
        self.counter += 1
        # Move left and bottom spines outward by 10 points
        ax.spines["left"].set_position(("outward", 10))
        ax.spines["bottom"].set_position(("outward", 10))
        # Hide the right and top spines
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        # Only show ticks on the left and bottom spines
        ax.yaxis.set_ticks_position("left")
        ax.xaxis.set_ticks_position("bottom")
        plt.show()

    def show(self):
        ret = self.gui.show()


if __name__ == "__main__":
    gui = MatplotGuiAutoUpdate()
    gui.show()
