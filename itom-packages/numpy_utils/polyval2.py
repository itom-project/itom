# coding=utf-8
import numpy as np


def polyval2(pol, koX, koY):
    x = koX.reshape(
        1, koX.size
    )  # max(koX.shape)) #These are columnVectors in difference to polyfitweighted function
    y = koY.reshape(1, koY.size)  # max(koY.shape))
    p = pol.reshape(1, pol.size)
    lx = max(x.shape)
    ly = max(y.shape)
    lp = max(p.shape)
    pts = lx
    n = np.int_((np.sqrt(1 + 8 * lp) - 3) / 2)

    if (min(p.shape) != 1) or (np.mod(n, 1) != 0) or (lx != ly):
        print(
            "polyval2:InvalidP",
            "P must be a vector of length (N+1)*(N+2)/2,",
            "where N is order. X and Y must be same size.",
        )
        return None
    # Construct Vandermonde matrix.
    V = np.zeros((pts, lp))
    V[:, 0] = np.ones(pts)
    ordercolumn = 1
    for order in range(1, n + 1):
        for ordercolumn in range(ordercolumn + 1, ordercolumn + order + 1):
            V[:, ordercolumn - 1] = x * V[:, ordercolumn - order - 1]
        ordercolumn += 1
        V[:, ordercolumn - 1] = y * V[:, ordercolumn - order - 2]
    # print('V=\n',V)
    z = np.dot(V, p.transpose())
    z = z.reshape(koX.shape)
    return z


##################end of polyval2##########################
