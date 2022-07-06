"""Plot 2.5D/ 3D
=============

The itom twipOGLFigure plot is for visualisation of 2D / 3D DataObjects and PointClouds.
"""

import numpy as np
from itom import plot25
# sphinx_gallery_thumbnail_path = '11_demos/_static/_thumb/demoPlotTwip.png'

###############################################################################
# A two dimensional sinus wave is generated und plotted using the TwipOGLFigure. 
N = 1024
amplitude = 128.0
periodes = 3

xs = np.linspace(-periodes*np.pi, periodes*np.pi, N)
ys = np.linspace(-periodes*np.pi, periodes*np.pi, N)

tau, phi = np.meshgrid(xs, ys) 

amp = np.sin(tau+phi) 


plot25(amp)

###############################################################################
# .. image:: ../../_static/demoPlotTwipOGL.png
#    :width: 100%