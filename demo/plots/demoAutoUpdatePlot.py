"""This demo shows two possibilities of how to create an auto updating plot, when the data does not come from a device allowing a live plot"""

from numpy import random
import time
from itom import *


def demo_autoUpdatePlot():
    d = dataObject.zeros([1, 3], "float64")
    # Option 1: open plot with fixed interval, the plot shares its values
    # from a given dataObject, update the dataObject regularily and call
    # the replot slot of the plot to force an update of the canvas (without that
    # slot, the canvas is updated once the user makes a zoom, clicks somewhere...)
    [i, h] = plot1(d, properties={"yAxisInterval": (0, 1)})

    t = time.time()
    for i in range(0, 50):
        d[0, :] = random.rand(3)
        h.call("replot")
        time.sleep(0.2)
    print("finished in %.2f s using replot" % (time.time() - t))

    # Option 2: similar to option 1, but the plot is continously given the same
    # object again as source. Some caching mechanism provides a quick replot
    # of the data. This option makes an automatic bounds-check of the new source
    # and can therefore automatically reset automatic axes intervals
    [i, h] = plot1(d)

    t = time.time()
    for i in range(0, 50):
        d[0, :] = random.rand(3)
        h["source"] = d
        time.sleep(0.2)
    print("finished in %.2f s using autoupdate" % (time.time() - t))


if __name__ == "__main__":
    demo_autoUpdatePlot()
