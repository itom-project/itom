"""1D animation
===============

This example demonstrates how matplotlib can be used to create
animated movie and export these in the mp4 movie format.
It is shown here with some random generated 2d images, which ware plotted via matplotlib.
By using the figure handle the animation is created. So you can plot your matplot
figures in your own way and used some similar syntax to create an animation.

First of all you must install the matplotlib package, e.g. from
https://pypi.org/project/matplotlib/

Then you must install the ffmpeg codec. A detailed description can be found on:
http://blog.gregzaal.com/how-to-install-ffmpeg-on-windows/

The build version of the ffmpeg codec can be downloaded here:
http://ffmpeg.zeranoe.com/builds/

Download and unzip the builds files to your harddrive. Typically the folder is like:
C:\\Program files\\ffmpeg

The bin folder of ffmpeg must be added to the path variables of your system:
C:\\Program files\\ffmpeg\\bin

Finally start the command prompt and run the command:
C:\\Program files\\ffmpeg\\bin\\ffmpeg.exe -codecs

or easier:
ffmpeg -codecs

"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

fig, ax = plt.subplots()

x = np.arange(0, 2 * np.pi, 0.01)
(line,) = ax.plot(x, np.sin(x))


def animate(i):
    line.set_ydata(np.sin(x + i / 50))  # update the data.
    return (line,)


ani = animation.FuncAnimation(fig, animate, interval=20, blit=True, save_count=50)

###############################################################################
# To save the animation, use e.g.
#
# .. code-block:: python
#
#     ani.save("movie.mp4")
#
# or
#
# .. code-block:: python
#
#     writer = animation.FFMpegWriter(
#     fps=15, metadata=dict(artist='Me'), bitrate=1800)
#     ani.save("movie.mp4", writer=writer)
#
# Please consider that this requires the ffmpeg installed on your computer.

plt.show()
