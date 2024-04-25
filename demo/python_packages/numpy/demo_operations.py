"""Operations
=============
"""

import numpy as np

# sphinx_gallery_thumbnail_path = '11_demos/_static/_thumb/demoNumpy.png'

###############################################################################
# **Arithmetic**
array = np.array([20, 30, 40, 50])
array2 = np.arange(4)
array2 - array

###############################################################################
array2**2

###############################################################################
10 * np.sin(array)

###############################################################################
array < 35

###############################################################################
# **Matrix operations**
#
# elementwise product
A = np.array([[1, 1], [0, 1]])
B = np.array([[2, 0], [3, 4]])
A * B

###############################################################################
# matrix product
A @ B
A.dot(B)

###############################################################################
# **Inline operation**
randVal = np.random.default_rng(1)
a = np.ones((2, 3), dtype=int)
b = randVal.random((2, 3))
a *= 3
a

###############################################################################
b += a
b

###############################################################################
# **Operations on all elements**
a = randVal.random((2, 3))
a.sum()

###############################################################################
a.min()

###############################################################################
a.mean()

###############################################################################
a.max()

###############################################################################
b = np.arange(12).reshape(3, 4)
b.sum(axis=0)  # sum of each column
b.min(axis=1)  # min of each row
b.cumsum(axis=1)  # cumulative sum along each row

###############################################################################
# **Univeral functions**
B = np.arange(3)
np.exp(B)

###############################################################################
np.sqrt(B)

###############################################################################
C = np.array([2.0, -1.0, 4.0])
np.add(B, C)
