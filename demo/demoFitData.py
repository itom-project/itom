import numpy as np

#polynomial of order 2 in x- and y-direction
def polyFuncOrder2x2(x, y):
    return 2.5 * x**2 + -1.7 * y**2 + 1.3 * x * y + 0.7 * x - 0.3 * y + 3.2
    
#vectorize this function to be evaluated over an array of x and y-coordinates
f = np.vectorize(polyFuncOrder2x2)

[X,Y] = np.meshgrid(np.arange(-10,10.5,0.5), np.arange(-10,10.5,0.5))
Z = f(X,Y)
total = np.prod(Z.shape)

#1. 2d polynomial fit with order x and order y = 2.
#Z_ must be a regular grid where the x- and y- values are
#defined by its axisScales and axisOffsets attributes
Z_ = dataObject(Z)
Z_.axisScales = (0.5, 0.5)
Z_.axisOffsets = (20, 20)

coeffs = filter("polyfitWeighted2D", Z_, 2, 2)

#2. reconstruct the fitted sphere using the determined coefficients.
#   -> create a data object with the desired size, scaling and offset for the
#       the grid of x- and y- values. The z-values are then calculated.
Z_reconstruction = Z_.copy()
Z_reconstruction[:,:] = float('nan')

filter("polyval2D", Z_reconstruction, coeffs, 2, 2)

#randomly select a number of samples unique values in the range [0,total)
samples = 100
randomUniqueValues = np.random.choice(total, samples)
X2 = dataObject([1, samples], 'float64')
Y2 = dataObject([1, samples], 'float64')
Z2 = dataObject([1, samples], 'float64')
c = Z.shape[1]

for i in range(0, samples):
    idx = randomUniqueValues[i]
    X2[0,i] = X[int(idx / c), idx % c]
    Y2[0,i] = Y[int(idx / c), idx % c]
    Z2[0,i] = Z[int(idx / c), idx % c]
    
#3. determine the polyonimal coefficients only using the random samples...
coeffs2 = filter("polyfitWeighted2DSinglePoints", X2, Y2, Z2, 2, 2)
#coeffs and coeffs2 must be the same!

#4. and reconstruct the entire surface for X and Y values
Z2_reconstruction = dataObject()
filter("polyval2DSinglePoints", dataObject(X), dataObject(Y), Z2_reconstruction, coeffs2, 2, 2)

sample_reconstruction = dataObject()
filter("polyval2DSinglePoints", X2, Y2, sample_reconstruction, coeffs2, 2, 2)
