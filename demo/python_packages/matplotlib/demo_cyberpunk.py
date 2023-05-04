"""Cyberpunk
==========

This `matplotlib <https://matplotlib.org/>`_ style `Cyberpunk <https://github.com/dhaitz/mplcyberpunk>`_ generate a futuristic style with neon light. 
In addition to `matplotlib <https://matplotlib.org/>`_, the style must be installed.

..  code-block:: bat
    
    pip install mplcyberpunk
"""

###############################################################################
# import namespaces
import numpy as np
import matplotlib.pyplot as plt
import mplcyberpunk


###############################################################################
# Generate demo x and y values.
x = np.linspace(0, 10, 20)
y = np.sin(x)
y2 = np.cos(x)


###############################################################################
# Change matplotlib.pyplot style and plot data.
with plt.style.context(['cyberpunk']):
    plt.figure(figsize = (6,6))
    plt.plot(x, y, marker='o', label='Line 1')
    plt.plot(x, y2, marker='x', label='Line 2')

    mplcyberpunk.make_lines_glow()
    mplcyberpunk.add_gradient_fill(alpha_gradientglow=0.5, gradient_start='zero')

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.grid()
    plt.show()