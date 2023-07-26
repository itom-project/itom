"""Shape manipulation
=====================
"""

import numpy as np

# sphinx_gallery_thumbnail_path = '11_demos/_static/_thumb/demoNumpy.png'

rg = np.random.default_rng(1)
a = np.floor(10 * rg.random((3, 4)))
a

###############################################################################
# return the array, flattened
a.ravel()

###############################################################################
# modified shape
a.reshape(6, 2)

###############################################################################
# transpose
a.T

###############################################################################
a.resize(2, 6)
