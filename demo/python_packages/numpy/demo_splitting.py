"""Splitting array
==================
"""

import numpy as np

# sphinx_gallery_thumbnail_path = '11_demos/_static/_thumb/demoNumpy.png'

###############################################################################
rg = np.random.default_rng(1)
a = np.floor(10 * rg.random((2, 12)))
a

###############################################################################
b = np.hsplit(a, 3)
b

###############################################################################
c = np.hsplit(a, (3, 4))
c
