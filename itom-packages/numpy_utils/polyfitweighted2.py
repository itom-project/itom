# coding=utf-8
# polyfitweighted2.py
# -------------------
#
# Find a least-squares fit of 2D data z(x,y) with an nth order
# polynomial, weighted by w(x,y) .
#
# By S.S. Rogers (2006)
#
# Usage
# ------
#
# P = polyfitweighted2(X,Y,Z,N,W) finds the coefficients of a polynomial
# P(X,Y) of degree N that fits the data Z best in a least-squares
# sense. P is a row vector of length (N+1)*(N+2)/2 containing the
# polynomial coefficients in ascending powers, 0th order first.
#
#   P = [p00 p10 p01 p20 p11 p02 p30 p21 p12 p03...]
#
# e.g. For a 3rd order fit,
# the regression problem is formulated in matrix format as:
#
#   wZ = V*P    or
#
#                      2       2   3   2     2      3
#   wZ = [w  wx  wy  wx  xy  wy  wx  wx y  wx y   wy ]  [p00
#                                                        p10
#                                                        p01
#                                                        p20
#                                                        p11
#                                                        p02
#                                                        p30
#                                                        p21
#                                                        p12
#                                                        p03]
#
# *Note:* P is not in the format of standard Matlab 1D polynomials. Use
# polval2.m to evaluate the polynomial in this format, at given values of
# x,y.
#
# X,Y must be 2D arrays, created for instance with meshgrid
# Z,W must be 2D arrays of size [length(X) length(Y)]
#
# based on polyfit.m by The Mathworks Inc. - see doc polyfit for more details
#
# Class support for inputs X,Y,Z,W:
#      float: double, single

import numpy as np
import numpy_utils.linalg_utils as nputils

##############################################################
def polyfitweighted2(koX, koY, koZ, n, margin=0.2, w=None):
    """
    polyfitweighted will try to solve a linear System,
    returning a description for an Object in the (n+1) Dimension
    into an inputted number of Points in 3Dimensions
    Input Values should already be masked.
    @param koX Matrix or Vector of type numpy array, describing XValue of Point
    @param koY Matrix or Vector of type numpy array, describing YValue of Point
    @param koZ Matrix or Vector of type numpy array, describing ZValue of Point, measured values
    @return Function will return a polynom in a row vector that best fits (RMS) Values in the (n+1) Dimension, so n=1 will do a Plane fit and n=2 a paraboloidfit
        also returns a Number, that shows the Quality of the fit, the Number of points that are in a margin around the fit
    Quality fit only works for n=1 or Planes at the moment
    """

    koX = koX.reshape(
        [koX.size, 1]
    )  # These are row Vectors in difference to ployval2 function
    koY = koY.reshape([koY.size, 1])  # size equals Number of Elements in array
    koZ = koZ.reshape([koZ.size, 1])
    if w == None:
        pw = np.ones((koX.size, 1))
    else:
        w = w.reshape([w.size, 1])
        pw = w.copy()
    if koX.shape[0] != koY.shape[0] != koZ.shape[0]:
        print(
            "polyfitweighted2:XYSizeMismatch\n X,Y,Z,W *must* be 1D arrays of same size"
        )

    # Hardcopys anlegen
    x = koX.copy()
    y = koY.copy()
    z = koZ.copy()
    ma = np.isfinite(z)
    x = x[
        ma
    ]  # Z-Werte werden gemessen, und koennen NaN oder infinite sein, solche Werte werden ausgefiltert
    y = y[ma]
    pw = pw[ma]
    z = z[ma]
    pts = z.size  # Anzahl gueltiger Punkte

    # Construct weighted Vandermonde matrix.
    V = np.zeros(
        (pts, np.int_((n + 1) * (n + 2) / 2))
    )  # Matrix wird angelegt/Reserviert
    V[:, 0] = pw
    ordercolumn = 1
    for order in range(1, n + 1):
        for ordercolumn in range(ordercolumn + 1, ordercolumn + order + 1):
            V[:, ordercolumn - 1] = x * V[:, ordercolumn - order - 1]
        ordercolumn += 1
        V[:, ordercolumn - 1] = y * V[:, ordercolumn - order - 2]

    # Solve least squares problem.
    [Q, R] = np.linalg.qr(V.astype("float32"))
    try:
        p = np.linalg.solve(R, np.dot(Q.conj().T, pw * z))
    except:
        print(
            "Hey das geht nicht! QR- Zerlegung failure, badly conditioned inputmatrix.You probably mismatched the input Vectors."
        )
        return None

    if R.shape[1] > R.shape[0]:
        print(
            "polyfitweighted2:PolyNotUnique Polynomial is not unique; degree >= number of data points."
        )
    # elif np.linalg.cond(R) > 1.0e10:
    elif (
        nputils.condest(R) > 1.0e10
    ):  # in Mathlab war das mal 'condest(R)' Warnung bezueglich Unterbestimmung, wenn gleiche Punkte in der Matrix auftauchen
        print(
            """  polyfitweighted2:RepeatedPointsOrRescale Polynomial is badly conditioned.
                            Remove repeated data points\nor try centering and scaling as described in HELP POLYFIT."""
        )

    # p = p[:,np.newaxis]
    p = p.transpose()
    # koX = koX.astype('float32')
    # koY = koY.astype('float32')

    # Check Quality of PolyFit
    nop = np.cumsum(np.absolute(p[0] + p[1] * koX + p[2] * koY - koZ) < margin)
    nop = nop[-1]

    return p, nop  # Polynomial coefficients are row vectors by convention.
