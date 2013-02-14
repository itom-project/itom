in order to use matplotlib with itom, copy the folder mpl_itom to PYTHONDIR\LIB\site_packages

and before using pylab, type the following lines in your python script:

import matplotlib
matplotlib.use('module://mpl_itom.backend_itomagg')

Please consider, that the command use raises a warning if executed after 

from pylab import *

This warning can be suppressed by 

matplotlib.use('...',False)