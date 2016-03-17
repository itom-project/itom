import matplotlib
#registeres itom as backend for matplotlib (important!)
#the false parameter indicates that no warning is emitted when the registering is called twice
matplotlib.use('module://mpl_itom.backend_itomagg',False)

import matplotlib.pyplot as plt
import numpy as np
x = np.random.randn(1000)
y = np.random.randn(1000) + 5

# normal distribution center at x=0 and y=5
plt.hist2d(x, y, bins=40)
plt.show()

#get current figure
current_figure = plt.gcf()

#set the keepSizeFixed property of the plot to true:
current_figure.canvas.manager.itomUI["keepSizeFixed"] = True
#alternative:
#plt.get_current_fig_manager().itomUI["keepSizeFixed"]

#change the size
current_figure.set_dpi(120)
current_figure.set_size_inches(5,5,True)
