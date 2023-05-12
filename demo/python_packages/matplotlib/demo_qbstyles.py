"""Quantum Black Styles
====================

This `matplotlib <https://matplotlib.org/>`_ style `Quantum Black Styles <https://github.com/quantumblacklabs/qbstyles>`_ generates figures in dark style. 
In addition to `matplotlib <https://matplotlib.org/>`_, the style must be installed.

..  code-block:: bat

    pip install qbstyles
"""

###############################################################################
# Import namespaces.
import numpy as np
import matplotlib.pyplot as plt
from qbstyles import mpl_style


###############################################################################
# Generate demo x and y values.
x = np.linspace(0, 10, 20)
y = np.sin(x)
y2 = np.cos(x)

###############################################################################
# Set dark style.
mpl_style(dark=True)

###############################################################################
# Plot data.
plt.figure(figsize = (6,6))
plt.plot(x, y, marker='o', label='Line 1')
plt.plot(x, y2, marker='x', label='Line 2')

plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.grid()
plt.show()