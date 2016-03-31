"""
This example demonstrates how matplotlib can be used to create 
animated movie and export these in the mp4 movie format. 
It is shown here with some random generated 2d images, which ware plotted via matplotlib. 
By using the figure handle the animation is created. So you can plot your matplot figures in your own way 
and used some similar syntax to create an animation. 

First of all you must install the matplotlib package:

.. http://www.lfd.uci.edu/~gohlke/pythonlibs/#matplotlib

Then you must install the ffmpeg codec. A detailed description can be found on: 

.. http://www.adaptivesamples.com/how-to-install-ffmpeg-on-windows

The build version of the ffmpeg codec can be downloaded here: 

.. https://ffmpeg.zeranoe.com/builds

Download and unzip the builds files to your harddrive. Typically the folder is like: 

.. C:\\Program files\\ffmpeg

The bin folder of ffmpeg must be added to the path variables of your system: 

.. C:\\Program files\\ffmpeg\\bin 

Finally start the command prompt and run the command: 

.. C:\\Proram files\\ffmpeg\\bin\\ffmpeg.exe -codecs

or easier: 

.. ffmpeg -codecs

"""

from itom import *
from itom import ui

import numpy as np
import matplotlib
#matplotlib.use('module://mpl_itom.backend_itomagg',False) # use this line to see the plot during the creation process of the animation
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as manimation
FFMpegWriter = manimation.writers['ffmpeg']

####### set value to create animation ##################################################
(value, accepted) = ui.getText("Animation title", "please type the name of the animation file", "animation")
comment = ""            # this comment will be added the mp4 file comments 
fps = 10                       # by variation of the fps parameter the speed of animation can be changed
dpi_plot = 120           # dpi of the plot
inches = 6                  # size of image
dpi_movie = 100        # dpi of the movie
numberImages = 100  # number of the images, which are created for the animation
##############################################################################

outputfile = value + ".mp4"
x = np.linspace(0, 2, 1000)


if accepted:
    fig = plt.figure()
    fig.set_dpi(dpi_plot)
    fig.set_size_inches(inches,inches,forward=True)
    ax = plt.axes(xlim=(0,2), ylim=(-2,2))
    line, = plt.plot([],[], lw = 2)
    
    metadata = dict(title=value, artist='Matplotlib', comment = comment)
    writer = FFMpegWriter(fps = fps, metadata=metadata, bitrate = -1, codec = 'mpeg4')
    writer.setup(fig, outputfile, dpi = dpi_movie)
    
    with writer.saving(fig, outputfile, dpi_movie):
        for cnt in range(0,numberImages):
            y = np.sin(2 * np.pi * (x - 0.02 * cnt))
            line.set_data(x, y)
            writer.grab_frame()
    
    print("animation finished")