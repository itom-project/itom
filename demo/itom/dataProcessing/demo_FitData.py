"""Fit data
===========

This demo shows how data fitting can be performed using the ``itom.dataObject`` and ``itom.algorithms``."""


import numpy as np
from itom import dataObject
from itom import algorithms
# sphinx_gallery_thumbnail_path = '11_demos/_static/_thumb/demoFitData.png'

###############################################################################
# Polynomial of order 2 in x- and y-direction
def polyFuncOrder2x2(x: float, y:float) -> float:
    return 2.5 * x ** 2 + -1.7 * y ** 2 + 1.3 * x * y + 0.7 * x - 0.3 * y + 3.2


###############################################################################
# Vectorize this function to be evaluated over an array of x and y-coordinates
f = np.vectorize(polyFuncOrder2x2)

[X, Y] = np.meshgrid(np.arange(-10, 10.5, 0.5), np.arange(-10, 10.5, 0.5))
Z = f(X, Y)
total = np.prod(Z.shape)

###############################################################################
# First create a 2d polynomial fit with order x = 2 and order y = 2.
# Z_ must be a regular grid where the x- and y- values are
# defined by its axisScales and axisOffsets attributes. 
Z_ = dataObject(Z)
Z_.axisScales = (0.5, 0.5)
Z_.axisOffsets = (20, 20)

coeffs = algorithms.polyfitWeighted2D(Z_, 2, 2)
print("coefficients: ", coeffs)

# Reconstruct the fitted sphere using the determined coefficients.
# First, create a ``dataObject`` with the desired size, scaling and offset for the
# the grid of x- and y- values. The z-values are then calculated.
Z_reconstruction = Z_.copy()
Z_reconstruction[:, :] = float("nan")

algorithms.polyval2D(Z_reconstruction, coeffs, 2, 2)

###############################################################################
# Randomly select a number of samples unique values in the range ``[0,total)``.
samples = 100
randomUniqueValues = np.random.choice(total, samples)
X2 = dataObject([1, samples], "float64")
Y2 = dataObject([1, samples], "float64")
Z2 = dataObject([1, samples], "float64")
c = Z.shape[1]

for i in range(samples):
    idx = randomUniqueValues[i]
    X2[0, i] = X[int(idx / c), idx % c]
    Y2[0, i] = Y[int(idx / c), idx % c]
    Z2[0, i] = Z[int(idx / c), idx % c]

###############################################################################
# Determine the polyonimal coefficients only using the random samples.
coeffs2 = algorithms.polyfitWeighted2DSinglePoints( X2, Y2, Z2, 2, 2)
# coeffs and coeffs2 must be the same!
print("fitted coefficient: ", coeffs2)

###############################################################################
# And reconstruct the entire surface for X and Y values.
Z2_reconstruction = dataObject()
algorithms.polyval2DSinglePoints(
    dataObject(X),
    dataObject(Y),
    Z2_reconstruction,
    coeffs2,
    2,
    2,
)

sample_reconstruction = dataObject()
algorithms.polyval2DSinglePoints(X2, Y2, sample_reconstruction, coeffs2, 2, 2)