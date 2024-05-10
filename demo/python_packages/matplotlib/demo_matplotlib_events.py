"""Events
=========

"""

import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots()
ax.plot(np.random.rand(10))


def onclick(event):
    print(
        "%s click: button=%d, x=%d, y=%d, xdata=%f, ydata=%f"
        % (
            "double" if event.dblclick else "single",
            event.button,
            event.x,
            event.y,
            event.xdata,
            event.ydata,
        )
    )


def onfigureenter(event):
    if event.x and event.y:
        print(
            "onfigureenter: x={:f}, y={:f}, inaxes={}".format(
                event.x, event.y, str(event.inaxes)
            )
        )
    else:
        print("onfigureenter: x=<None>, y=<None>, inaxes=%s" % str(event.inaxes))


cid = fig.canvas.mpl_connect("button_press_event", onclick)
fig.canvas.mpl_connect("figure_enter_event", onfigureenter)
