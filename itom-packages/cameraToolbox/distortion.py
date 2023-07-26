import numpy as np
from scipy.optimize import leastsq, minimize
from scipy import linalg as la
from itom import dataObject, filter
import warnings

"""Toolbox for distort and undistort point coordinates as well as determine the radial distortion"""

#######################################################
def getPointGrid(shape, pitch, center, rotation=0, repr="2d"):
    """creates a dataObject with coordinates of points on a regular grid.

    Returns a [m,n,2] array if repr is '3d', else a [m*n,2] array
    with the coordinates, where the last dimensions contains the x- and y-coordinate
    of each point respectively.

    * shape = (num_rows, num_cols)
    * pitch = pixels between each point in one row or column (pitchX, pitchY)
    * center = (centerX, centerY)
    * rotation = possible rotation of grid in degree
    * repr = '2d' or '3d'. See return above.
    """
    xSpots, ySpots = np.meshgrid(
        (np.arange(shape[1]) - (shape[1] - 1) / 2.0) * pitch[1],
        (np.arange(shape[0]) - (shape[0] - 1) / 2.0) * pitch[0],
    )
    theta = rotation / 180.0 * np.pi
    xSpots = xSpots * np.cos(theta) - ySpots * np.sin(theta) + center[0]
    ySpots = xSpots * np.sin(theta) + ySpots * np.cos(theta) + center[1]

    if repr.lower() == "3d":
        return dataObject(np.dstack([xSpots, ySpots]))
    else:
        num = shape[0] * shape[1]
        return dataObject(np.vstack([xSpots.reshape(num), ySpots.reshape(num)]))


#######################################################
def getPointGridDistorted(
    shape, pitch, center, rotation=0, k1=0, k2=0, k3=0, repr="2d"
):
    """creates a dataObject with coordinates of distorted points on a regular grid.

    Returns a [m,n,2] array if repr is '3d', else a [m*n,2] array
    with the coordinates, where the last dimensions contains the x- and y-coordinate
    of each point respectively.

    The relationship between undistorted coordinates (x_u, y_u)
    and their distorted correspondences (x_d, y_d) are given
    by OpenCV:

    x_u = x_d * (1 + k1 * r^2 + k2 * r^4 + k3 * r^6)
    y_u = y_d * (1 + k1 * r^2 + k2 * r^4 + k3 * r^6)

    where r = sqrt((x_u - center[0])^2 + (y_u - center[1])^2).

    * shape = (num_rows, num_cols)
    * pitch = pixels between each point in one row or column
    * center = (centerX, centerY)
    * rotation = possible rotation of grid in degree
    * repr = '2d' or '3d'. See return above.
    """
    xSpots, ySpots = np.meshgrid(
        (np.arange(shape[1]) - (shape[1] - 1) / 2.0) * pitch[1],
        (np.arange(shape[0]) - (shape[0] - 1) / 2.0) * pitch[0],
    )
    theta = rotation / 180.0 * np.pi
    xTemp = xSpots * np.cos(theta) - ySpots * np.sin(theta)
    yTemp = xSpots * np.sin(theta) + ySpots * np.cos(theta)
    r2 = xTemp * xTemp + yTemp * yTemp
    factor = k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2 + 1
    xSpots = xTemp * factor + center[0]
    ySpots = yTemp * factor + center[1]

    if repr.lower() == "3d":
        return dataObject(np.dstack([xSpots, ySpots]))
    else:
        num = shape[0] * shape[1]
        return dataObject(np.vstack([xSpots.reshape(num), ySpots.reshape(num)]))


#######################################################
def getPointGridRotated(shape, pitch, center, rotation=0, repr="2d"):
    """creates a dataObject with coordinates of shifted, scaled and rotated points on a regular grid.

    Returns a [m,n,2] array if repr is '3d', else a [m*n,2] array
    with the coordinates, where the last dimensions contains the x- and y-coordinate
    of each point respectively.

    * shape = (num_rows, num_cols)
    * pitch = pixels between each point in one row or column (pitchX, pitchY)
    * center = (centerX, centerY)
    * rotation = possible rotation of grid in degree
    * repr = '2d' or '3d'. See return above.
    """
    xSpots, ySpots = np.meshgrid(
        (np.arange(shape[1]) - (shape[1] - 1) / 2.0) * pitch[1],
        (np.arange(shape[0]) - (shape[0] - 1) / 2.0) * pitch[0],
    )
    theta = rotation / 180.0 * np.pi
    xTemp = xSpots * np.cos(theta) - ySpots * np.sin(theta)
    yTemp = xSpots * np.sin(theta) + ySpots * np.cos(theta)
    xSpots = xTemp + center[0]
    ySpots = yTemp + center[1]

    if repr.lower() == "3d":
        return dataObject(np.dstack([xSpots, ySpots]))
    else:
        num = shape[0] * shape[1]
        return dataObject(np.vstack([xSpots.reshape(num), ySpots.reshape(num)]))


#######################################################
def undistortPointGrid(pointsDistorted, rows, cols, coeffs, repr="2d"):
    """
    Undistort point coordinates based on the given coefficients 'coeffs'.

    This method does not de-rotate or shift the points, but only correct
    the distortion. The relationship between distorted (x_d, y_d) and
    undistorted points (x_u, y_u) is given by the following equation (based on OpenCV
    definition):

    x_d = x_u * (1 + k1 * r^2 + k2 * r^4 + k3 * r^6)
    y_d = y_u * (1 + k1 * r^2 + k2 * r^4 + k3 * r^6)

    The radius is hereby

    r = sqrt((x_u - centerX)^2 + (y_u - centerY)^2).

    In order to solve for x_u and y_u, the undistortion is calculated by
    an iterative approach where the radius is estimated based on the
    distorted coordinates and then approaches the real value (see OpenCV:: cvUndistortPoints)
    """
    if len(coeffs) == 5 or len(coeffs) == 4:
        return points
    elif len(coeffs) != 8:
        raise ValueError(
            "coeffs must have 4 or 5 (no distortion) or 8 (distortion) components"
        )

    pitchX, pitchY, centerX, centerY, rotation, k1, k2, k3 = coeffs

    if pointsDistorted.ndim == 3:
        x_d = np.array(pointsDistorted[:, :, 0]).reshape([rows * cols])
        y_d = np.array(pointsDistorted[:, :, 1]).reshape([rows * cols])
    else:
        x_d = np.array(pointsDistorted[:, 0]).reshape([rows * cols])
        y_d = np.array(pointsDistorted[:, 1]).reshape([rows * cols])

    x_u = x_u0 = x_d - centerX
    y_u = y_u0 = y_d - centerY

    # based on the OpenCV algorithm cvUndistortPoints, the distortion correction
    # is done with an iterative approach.
    r2old = x_u * 0
    error = 1e12
    maxIter = 100
    iter = 0

    while error >= 1e-6 and iter < maxIter:
        r2 = x_u * x_u + y_u * y_u
        error = np.nanmean(np.abs(r2 - r2old))
        iter += 1
        factor = 1.0 / (1 + r2 * (k1 + r2 * (k2 + k3 * r2)))
        x_u = x_u0 * factor
        y_u = y_u0 * factor
        r2old = r2

    if iter >= maxIter:
        warnings.warn(
            "The iterative undistortion algorithm did not converge within the maximum number of iterations.",
            DeprecationWarning,
        )

    theta = -rotation / 180.0 * np.pi
    x_u_derotated = x_u * np.cos(theta) - y_u * np.sin(theta)
    y_u_derotated = x_u * np.sin(theta) + y_u * np.cos(theta)

    x_u_derotated += centerX
    y_u_derotated += centerY

    if repr.lower() == "3d":
        return dataObject(np.dstack([x_u_derotated, y_u_derotated]))
    else:
        num = shape[0] * shape[1]
        return dataObject(
            np.vstack([x_u_derotated.reshape(num), y_u_derotated.reshape(num)])
        )


#######################################################
def drawPointGrid(points, rows, cols, canvas=None, scale=1.0):
    """draw the point coordinates in a given canvas (or a newly created canvas if
    canvas == None) using the OpenCV method 'cvDrawChessboardCorners'

    * points must be a [m,n,2] or [m*n,2] dataObject
    * rows = m
    * cols = n
    * canvas: rgba32 dataObject or None if a new canvas should be created

    Returns the given or newly created canvas as dataObject (type: rgba32)
    """
    if points.ndim == 3:
        points_ = dataObject(points).reshape([rows * cols, 2])
    else:
        points_ = dataObject(points)

    points_ *= scale

    newCanvas = True
    if canvas and type(canvas) is dataObject and canvas.dtype == "rgba32":
        newCanvas = False

    if newCanvas:
        minH = int(min(points_[:, 1])) - 50
        maxH = min(10000, int(max(points_[:, 1]) + 50))
        minW = int(min(points_[:, 0])) - 50
        maxW = min(10000, int(max(points_[:, 0]) + 50))
        canvas = dataObject.zeros([maxH - minH, maxW - minW], "rgba32")
        points_[:, 1] -= minH
        points_[:, 0] -= minW
    else:
        minH = minW = 0

    filter("cvDrawChessboardCorners", canvas, (cols, rows), points_, 1)
    canvas.axisOffsets = (-minH, -minW)

    return canvas


#######################################################
def createDistortionMap(coeffs, points, rows, cols):
    """
    returns a float64 dataObject with the evaluated distortion
    based on the given coefficients.
    """
    if points.ndim == 3:
        points_ = dataObject(points).reshape([rows * cols, 2])
    else:
        points_ = dataObject(points)

    minH = int(min(points_[:, 1])) - 50
    maxH = min(10000, int(max(points_[:, 1]) + 50))
    minW = int(min(points_[:, 0])) - 50
    maxW = min(10000, int(max(points_[:, 0]) + 50))
    width = maxW - minW
    height = maxH - minH

    [X, Y] = np.meshgrid(range(minW, maxW + 1), range(minH, maxH - 1))
    X = X.astype("float64") - coeffs[2]
    Y = Y.astype("float64") - coeffs[3]
    radius2 = X * X + Y * Y
    distortionMap = (
        dataObject(
            coeffs[5] * radius2
            + coeffs[6] * radius2 * radius2
            + coeffs[7] * radius2 * radius2 * radius2
        )
        * 100.0
    )
    distortionMap.valueDescription = "Distortion"
    distortionMap.valueUnit = "%"
    distortionMap.axisOffsets = (-minH, -minW)

    return distortionMap


#######################################################
def getMeanDistance(x1, y1, x2, y2):
    """returns the mean distance between all corresponding points.

    Each pair of points is given by its (x1,y1) and (x2,y2) value.
    """
    return np.nanmean(np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2))


#######################################################
def meritFitGrid(params, xe, ye):
    """merit function for minimize-function that calculates the
    current mean distance between the given grid of points (xe,ye)
    and the calculated one based on the current set of parameters.

    This implementation does not consider any distortion, but the rotation.
    """
    pitchX, pitchY, centerX, centerY, rotation = params
    xy_grid = getPointGrid(
        xe.shape, [pitchX, pitchY], [centerX, centerY], rotation, "3d"
    )
    dist = getMeanDistance(xy_grid[:, :, 0], xy_grid[:, :, 1], xe, ye)
    # print(params, dist)
    return dist


#######################################################
def meritFitGridDistortion(params, xe, ye):
    """merit function for minimize-function that calculates the
    current mean distance between the given grid of points (xe,ye)
    and the calculated one based on the current set of parameters.

    This implementation considers shift, scaling and distortion (no rotation).
    """
    pitchX, pitchY, centerX, centerY, k1, k2, k3 = params
    xy_grid = getPointGridDistorted(
        xe.shape, [pitchX, pitchY], [centerX, centerY], 0, k1, k2, k3, "3d"
    )
    dist = getMeanDistance(xy_grid[:, :, 0], xy_grid[:, :, 1], xe, ye)
    # print(params, dist)
    return dist


#######################################################
def meritFitGridDistortionRotation(params, xe, ye):
    """merit function for minimize-function that calculates the
    current mean distance between the given grid of points (xe,ye)
    and the calculated one based on the current set of parameters.

    This implementation considers shift, scaling, rotation and distortion.
    """
    pitchX, pitchY, centerX, centerY, rotation, k1, k2, k3 = params
    xy_grid = getPointGridDistorted(
        xe.shape, [pitchX, pitchY], [centerX, centerY], rotation, k1, k2, k3, "3d",
    )
    dist = getMeanDistance(xy_grid[:, :, 0], xy_grid[:, :, 1], xe, ye)
    # print(params, dist)
    return dist


#######################################################
def meritFitGridRotation(params, xe, ye):
    """merit function for minimize-function that calculates the
    current mean distance between the given grid of points (xe,ye)
    and the calculated one based on the current set of parameters.

    This implementation considers only shift, scaling and rotation.
    """
    pitchX, pitchY, centerX, centerY, rotation = params
    xy_grid = getPointGridRotated(
        xe.shape, [pitchX, pitchY], [centerX, centerY], rotation, "3d"
    )
    dist = getMeanDistance(xy_grid[:, :, 0], xy_grid[:, :, 1], xe, ye)
    # print(params, dist)
    return dist


#######################################################
def estimateRotationByPCA(xe, ye, rows, cols):
    """
    apply principal component analysis to points to get a coarse
    estimate for the rotation angle
    """
    data = np.hstack(
        [
            np.array(xe).reshape([rows * cols, 1]),
            np.array(ye).reshape([rows * cols, 1]),
        ]
    )
    mask = np.isfinite(data)[:, 0]
    data = data[mask, :]
    m, n = data.shape
    # mean center the data
    data -= data.mean(axis=0)
    # calculate the covariance matrix
    R = np.cov(data, rowvar=False)
    # calculate eigenvectors & eigenvalues of the covariance matrix
    # use 'eigh' rather than 'eig' since R is symmetric,
    # the performance gain is substantial
    evals, evecs = la.eigh(R)
    # sort eigenvalue in decreasing order
    idx = np.argsort(evals)[::-1]
    evecs = evecs[:, idx]
    # sort eigenvectors according to same index
    evals = evals[idx]

    if cols >= rows:
        # the first evecs is hopefully the one in x-direction
        dy = evecs[1, 0]
        dx = evecs[0, 0]
        if dx < 0:
            dx *= -1
            dy *= -1
        rotation = np.degrees(np.arctan2(dy, dx))
    else:
        # the first evecs is hopefully the one in y-direction
        dy = evecs[1, 0]
        dx = evecs[0, 0]
        if dy < 0:
            dx *= -1
            dy *= -1
        rotation = np.degrees(np.arctan2(dx, dy))
    # return the eigenvalues, and eigenvectors
    return rotation


#######################################################
def guessInitialParameters(
    pointGrid, rows, cols, withDistortion=False, withRotation=True
):
    """gives an initial guess for the optimization parameters:

    withDistortion == False, withRotation == False:
        x0 = [grid-pitch, centerX, centerY]
    withDistortion == False, withRotation == True:
        x0 = [grid-pitch, centerX, centerY, rotation]
    withDistortion == True, withRotation == True:
        x0 = [grid-pitch, centerX, centerY, rotation, k1, k2, k3]
    """
    if pointGrid.ndim == 2:
        xe = pointGrid[:, 0]
        ye = pointGrid[:, 1]
    else:
        xe = pointGrid[:, :, 0]
        ye = pointGrid[:, :, 1]

    if withRotation:
        rotation = estimateRotationByPCA(xe, ye, rows, cols)
    else:
        rotation = 0

    dx = max(xe) - min(xe)
    dy = max(ye) - min(ye)
    pitchX = dx / cols
    pitchY = dy / rows
    centerX = min(xe) + 0.5 * dx
    centerY = max(xe) + 0.5 * dy

    if withDistortion and withRotation:
        return [pitchX, pitchY, centerX, centerY, rotation, 0, 0, 0]
    elif withDistortion:
        return [pitchX, pitchY, centerX, centerY, 0, 0, 0]
    elif withRotation:
        return [pitchX, pitchY, centerX, centerY, rotation]


####################################################
def fitGrid(
    distortedPointGrid,
    rows=None,
    cols=None,
    x0=None,
    withDistortion=False,
    withRotation=True,
):
    """
    main function to start the fit for a regular or distorted grid of points to the
    given set of distorted points.

    Parameters
    -------------
    * distortedPointGrid : dataObject, np.array:
        either a [Mx2] or [MxNx2] dataObject with the distorted point coordinates (x,y) along the 2-dim-axis
    * rows : int
        number of rows (necessary if distortedPointGrid has shape [Mx2])
    * cols : int
        number of cols (necessary if distortedPointGrid has shape [Mx2])
    * x0 : list
        initial optimization parameters (same order and meaning than resulting list). If not given, they
        are estimated using the method 'guessInitialParameters'.
    * withDistortion : bool
        defines if radial distortion coefficients should be optimized, too

    Returns
    --------
    * list of coefficients:
        the length of this list depends on the type of optimization and the length of the input data.
        If x0 contains 7 parameters (pitch, centerX, centerY, rotation, k1, k2, k3), all these 7 values
        are returned, too. All components, that have not been released for optimization during this call
        remain unchanged in the vector. If distortion should be optimized, but x0 only contains 3 or 4 values,
        a list with 7 elements is returned.
    * result dictionary of scipy.optimize.minimize function that is used for the non-linear optimization
    """
    if not rows:
        if distortedPointGrid.ndims == 2:
            raise RuntimeError(
                "distortedPointGrid must have 3 dimensions to guess parameters 'rows'"
            )
        rows = distortedPointGrid.shape[0]

    if not cols:
        if distortedPointGrid.ndims == 2:
            raise RuntimeError(
                "distortedPointGrid must have 3 dimensions to guess parameters 'cols'"
            )
        cols = distortedPointGrid.shape[1]

    if not x0:
        x0 = guessInitialParameters(
            distortedPointGrid, rows, cols, withDistortion, withRotation
        )

    if distortedPointGrid.ndim == 2:
        xe = distortedPointGrid[:, 0].astype("float64")
        ye = distortedPointGrid[:, 1].astype("float64")
    else:
        xe = distortedPointGrid[:, :, 0].astype("float64")
        ye = distortedPointGrid[:, :, 1].astype("float64")

    if withDistortion and withRotation:
        # optimize scaling, rotation and distortion
        if len(x0) == 5:
            x0 = x0 + [0, 0, 0]
        elif len(x0) == 4:
            x0 = x0 + [0, 0, 0, 0]
        coeffs = minimize(
            meritFitGridDistortionRotation, x0, args=(xe, ye), method="Nelder-Mead",
        )
        result = coeffs.x

    elif withDistortion:
        # optimize scaling and distortion
        if len(x0) == 5:
            rotation = x0[4]
            x1 = [*x0[0:4], 0, 0, 0]
            coeffs = minimize(
                meritFitGridDistortion, x1, args=(xe, ye), method="Nelder-Mead"
            )
            result = [*coeffs.x[0:4], rotation, *coeffs.x[4:7]]
        elif len(x0) == 4:
            x0 = x0 + [0, 0, 0]
            coeffs = minimize(
                meritFitGridDistortion, x0, args=(xe, ye), method="Nelder-Mead"
            )
            result = [*coeffs.x[0:4], 0.0, *coeffs.x[4:7]]
        elif len(x0) == 8:
            coeffs = minimize(
                meritFitGridDistortion, x0, args=(xe, ye), method="Nelder-Mead"
            )
            result = coeffs.x

    elif withRotation:
        # optimize scaling and rotation
        if len(x0) == 5:
            coeffs = minimize(
                meritFitGridRotation, x0, args=(xe, ye), method="Nelder-Mead"
            )
            result = [*coeffs.x]
        elif len(x0) == 4:
            x0 = x0 + [
                0,
            ]
            coeffs = minimize(
                meritFitGridRotation, x0, args=(xe, ye), method="Nelder-Mead"
            )
            result = [*coeffs.x[0:4]]
        elif len(x0) == 8:
            x1 = x0[0:5]
            distortion = x0[5:8]
            coeffs = minimize(
                meritFitGridRotation, x0, args=(xe, ye), method="Nelder-Mead"
            )
            result = [*coeffs.x] + distortion

    else:
        # optimize scaling
        x1 = x0[0:4]
        coeffs = minimize(meritFitGrid, x1, args=(xe, ye), method="Nelder-Mead")
        result = x0
        result[0:4] = coeffs.x[0:4]

    return result, coeffs
