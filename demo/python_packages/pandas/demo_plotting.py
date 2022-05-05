"""Plotting
===========

"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

###############################################################################
plt.figure()
timeStemps = pd.Series(np.random.randn(1000), index=pd.date_range("1/1/2000", periods=1000))
timeStemps = timeStemps.cumsum()
timeStemps.plot()
plt.show()

###############################################################################
dataFrame = pd.DataFrame(np.random.randn(1000, 4), index=timeStemps.index, columns=["A", "B", "C", "D"])
dataFrame = dataFrame.cumsum()
plt.figure()
dataFrame.plot()
plt.legend(loc="best")
