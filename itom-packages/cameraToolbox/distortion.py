import numpy as np
from scipy.optimize import leastsq, minimize
from scipy import linalg as la
from itom import dataObject, filter
import warnings

'''Toolbox for distort and undistort point coordinates as well as determine the radial distortion'''

#######################################################
def getPointGrid(shape, pitch, center, rotation=0, repr = '2d'):
    '''creates a dataObject with coordinates of points on a regular grid.
    
    Returns a [m,n,2] array if repr is '3d', else a [m*n,2] array
    with the coordinates, where the last dimensions contains the x- and y-coordinate
    of each point respectively.
    
    * shape = (num_rows, num_cols)
    * pitch = pixels between each point in one row or column
    * center = (centerX, centerY)
    * rotation = possible rotation of grid in degree
    * repr = '2d' or '3d'. See return above.
    '''
    xSpots, ySpots = np.meshgrid(
             (np.arange(shape[1]) - (shape[1]-1)/2.)*pitch, 
             (np.arange(shape[0]) - (shape[0]-1)/2.)*pitch)
    theta = rotation/180.*np.pi
    xSpots = xSpots * np.cos(theta) - ySpots * np.sin(theta) + center[0]
    ySpots = xSpots * np.sin(theta) + ySpots * np.cos(theta) + center[1]
    
    if repr.lower() == '3d':
        return dataObject(np.dstack([xSpots, ySpots]))
    else:
        num = shape[0] * shape[1]
        return dataObject(np.vstack([xSpots.reshape(num), ySpots.reshape(num)]))
        
#######################################################
def getPointGridDistorted(shape, pitch, center, rotation=0, k1 = 0, k2 = 0, k3 = 0, repr = '2d'):
    '''creates a dataObject with coordinates of distorted points on a regular grid.
    
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
    '''
    xSpots, ySpots = np.meshgrid(
             (np.arange(shape[1]) - (shape[1]-1)/2.)*pitch, 
             (np.arange(shape[0]) - (shape[0]-1)/2.)*pitch)
    theta = rotation/180.*np.pi
    xTemp = xSpots*np.cos(theta) - ySpots*np.sin(theta)
    yTemp = xSpots*np.sin(theta) + ySpots*np.cos(theta)
    r2 = xTemp * xTemp + yTemp * yTemp
    factor = k1 * r2 + k2 * r2 * r2 + \
                k3 * r2 * r2 * r2 + 1
    xSpots = xTemp * factor + center[0]
    ySpots = yTemp * factor + center[1]
    
    if repr.lower() == '3d':
        return dataObject(np.dstack([xSpots, ySpots]))
    else:
        num = shape[0] * shape[1]
        return dataObject(np.vstack([xSpots.reshape(num), ySpots.reshape(num)]))
        
#######################################################
def undistortPointGrid(pointsDistorted, rows, cols, coeffs, repr = '2d'):
    '''
    x_d = x_u * (1 + k1 * r^2 + k2 * r^4 + k3 * r^6)
    y_d = y_u * (1 + k1 * r^2 + k2 * r^4 + k3 * r^6)
    '''
    if len(coeffs) == 4:
        return points
    elif len(coeffs) != 7:
        raise ValueError("coeffs must have 4 (no distortion) or 7 (distortion) components")
    
    pitch, centerX, centerY, rotation, k1, k2, k3 = coeffs
    
    if pointsDistorted.ndim == 3:
        x_d = np.array(pointsDistorted[:,:,0]).reshape([rows*cols])
        y_d = np.array(pointsDistorted[:,:,1]).reshape([rows*cols])
    else:
        x_d = np.array(pointsDistorted[:,0]).reshape([rows*cols])
        y_d = np.array(pointsDistorted[:,1]).reshape([rows*cols])
    
    x_u = x_u0 = x_d -  centerX
    y_u = y_u0 = y_d -  centerY
    
    #based on the OpenCV algorithm cvUndistortPoints, the distortion correction
    #is done with an iterative approach.
    r2old = x_u * 0
    error = 1e12
    maxIter = 100
    iter = 0
    
    while (error >= 1e-6 and iter < maxIter):
        r2 = x_u * x_u + y_u * y_u
        error = np.mean(np.abs(r2 - r2old))
        iter += 1
        factor = 1.0 / (1 + r2 * (k1 + r2 * (k2 + k3 * r2)))
        x_u = x_u0 * factor
        y_u = y_u0 * factor
        r2old = r2
    
    if iter >= maxIter:
        warnings.warn("The iterative undistortion algorithm did not converge within the maximum number of iterations.", DeprecationWarning)
    
    theta = -rotation/180.*np.pi
    x_u_derotated = x_u*np.cos(theta) - y_u*np.sin(theta)
    y_u_derotated = x_u*np.sin(theta) + y_u*np.cos(theta)
    
    x_u_derotated += centerX
    y_u_derotated += centerY
    
    if repr.lower() == '3d':
        return dataObject(np.dstack([x_u_derotated, y_u_derotated]))
    else:
        num = shape[0] * shape[1]
        return dataObject(np.vstack([x_u_derotated.reshape(num), y_u_derotated.reshape(num)]))
    

#######################################################
def drawPointGrid(points, rows, cols, canvas = None):
    '''draw the point coordinates in a given canvas (or a newly created canvas if
    canvas == None) using the OpenCV method 'cvDrawChessboardCorners'
    
    * points must be a [m,n,2] or [m*n,2] dataObject
    * rows = m
    * cols = n
    * canvas: rgba32 dataObject or None if a new canvas should be created
    
    Returns the given or newly created canvas as dataObject (type: rgba32)
    '''
    if points.ndim == 3:
        points_ = dataObject(points).reshape([rows*cols,2])
    else:
        points_ = dataObject(points)
        
    newCanvas = True
    if canvas and type(canvas) is dataObject and canvas.dtype=='rgba32':
        newCanvas = False
    
    if newCanvas:
        height = min(10000, int(max(points_[:,1]) + 50))
        width = min(10000, int(max(points_[:,0]) + 50))
        canvas = dataObject.zeros([height, width], 'rgba32')
    
    filter("cvDrawChessboardCorners", canvas, (cols, rows), points_, 1)
    return canvas
    
#######################################################
def createDistortionMap(coeffs, points, rows, cols):
    '''
    returns a float64 dataObject with the evaluated distortion
    based on the given coefficients.
    '''
    if points.ndim == 3:
        points_ = dataObject(points).reshape([rows*cols,2])
    else:
        points_ = dataObject(points)
    
    height = min(10000, int(max(points_[:,1]) + 50))
    width = min(10000, int(max(points_[:,0]) + 50))
    
    [X,Y] = np.meshgrid(range(width), range(height))
    X = X.astype('float64') - coeffs[1]
    Y = Y.astype('float64') - coeffs[2]
    radius2 = X*X + Y*Y
    distortionMap = dataObject(coeffs[4] * radius2 + \
                   coeffs[5] * radius2 * radius2 + coeffs[6] * radius2 * radius2 * radius2) * 100.0
    distortionMap.valueDescription = "Distortion"
    distortionMap.valueUnit = "%"

    return distortionMap
    
    
#######################################################
def getMeanDistance(x1, y1, x2, y2):
    '''returns the mean distance between all corresponding points.
    
    Each pair of points is given by its (x1,y1) and (x2,y2) value.
    '''
    return np.sqrt((x1 - x2)**2 + (y1 - y2)**2).mean()
    
#######################################################
def meritFitGrid(params, xe, ye):
    '''merit function for minimize-function that calculates the
    current mean distance between the given grid of points (xe,ye)
    and the calculated one based on the current set of parameters.
    
    This implementation does not consider any distortion, but the rotation.
    '''
    pitch, centerX, centerY, rotation = params
    xy_grid = getPointGrid(xe.shape, pitch, [centerX, centerY], rotation, '3d')
    dist = getMeanDistance(xy_grid[:,:,0], xy_grid[:,:,1], xe, ye)
    #print(params, dist)
    return dist

#######################################################
def meritFitGridDistortion(params, xe, ye):
    '''merit function for minimize-function that calculates the
    current mean distance between the given grid of points (xe,ye)
    and the calculated one based on the current set of parameters.
    
    This implementation considers any rotation and distortion.
    '''
    pitch, centerX, centerY, rotation, k1, k2, k3 = params
    xy_grid = getPointGridDistorted(xe.shape, pitch, [centerX, centerY], rotation, k1, k2, k3, '3d')
    dist = getMeanDistance(xy_grid[:,:,0], xy_grid[:,:,1], xe, ye)
    #print(params, dist)
    return dist
    
#######################################################
def estimateRotationByPCA(xe, ye, rows, cols):
    """
    apply principal component analysis to points to get a coarse
    estimate for the rotation angle
    """
    data = np.hstack([np.array(xe).reshape([rows*cols,1]), np.array(ye).reshape([rows*cols,1])])
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
    evecs = evecs[:,idx]
    # sort eigenvectors according to same index
    evals = evals[idx]
    
    if (cols >= rows):
        # the first evecs is hopefully the one in x-direction
        dy = evecs[1,0]
        dx = evecs[0,0]
        if dx < 0:
            dx *= -1
            dy *= -1
        rotation = np.degrees(np.arctan2(dy,dx))
    else:
        #the first evecs is hopefully the one in y-direction
        dy = evecs[1,0]
        dx = evecs[0,0]
        if dy < 0:
            dx *= -1
            dy *= -1
        rotation = np.degrees(np.arctan2(dx,dy))
    # return the eigenvalues, and eigenvectors
    return rotation
    
#######################################################
def guessInitialParameters(pointGrid, rows, cols, withDistortion = False, withRotation = True):
    '''gives an initial guess for the optimization parameters:
    
    withDistortion == False:
        x0 = [grid-pitch, centerX, centerY, rotation]
    withDistortion == True:
        x0 = [grid-pitch, centerX, centerY, rotation, k1, k2, k3]
    '''
    if pointGrid.ndim == 2:
        xe = pointGrid[:,0]
        ye = pointGrid[:,1]
    else:
        xe = pointGrid[:,:,0]
        ye = pointGrid[:,:,1]
        
    if withRotation:
        rotation = estimateRotationByPCA(xe,ye,rows,cols)
    else:
        rotation = 0
    
    dx = max(xe)-min(xe)
    dy = max(ye)-min(ye)
    pitch = np.mean([dx / cols, dy / rows])
    centerX = min(xe) + 0.5 * dx
    centerY = max(xe) + 0.5 * dy
    
    if not withDistortion:
        return [pitch, centerX, centerY, rotation]
    else:
        return [pitch, centerX, centerY, rotation, 0, 0, 0]
    
####################################################
def expandInitialParametersByDistortion(coeffs):
    '''if the coefficients are given without distortion components
    and the 'fitGrid' method should be called again with a consideration
    of the distortion, use this method to expand the coefficients
    to a larger list containing k1, k2 and k3 (initialized to 0)
    '''
    if len(coeffs) != 7:
        return list(coeffs) + [0,] * (7 - len(coeffs))
    
####################################################
def fitGrid(distortedPointGrid, rows = None, cols = None, x0 = None, withDistortion = False):
    '''
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
        are estimated using the method 'guessInitialParameters'
    * withDistortion : bool
        defines if radial distortion coefficients should be optimized, too
    
    Returns
    --------
    * list of coefficients:
          either [pitch, centerX, centerY, rotation] (if not withDistortion)
          or [pitch, centerX, centerY, rotation, k1, k2, k3] (if withDistortion)
    * result dictionary of scipy.optimize.minimize function that is used for the non-linear optimization
    '''
    if not rows:
        if distortedPointGrid.ndims == 2:
            raise RuntimeError("distortedPointGrid must have 3 dimensions to guess parameters 'rows'")
        rows = distortedPointGrid.shape[0]
    
    if not cols:
        if distortedPointGrid.ndims == 2:
            raise RuntimeError("distortedPointGrid must have 3 dimensions to guess parameters 'cols'")
        cols = distortedPointGrid.shape[1]
    
    if not x0:
        x0 = guessInitialParameters(distortedPointGrid, rows, cols, withDistortion)
        
    if withDistortion and len(x0) != 7:
        raise ValueError("x0 must have 7 elements if distortion should be optimized, too: [pitch, centerX, centerY, rotation, k2, k4, k6]")
    elif not withDistortion and len(x0) != 4:
        raise ValueError("x0 must have 4 elements if distortion should not be optimized: [pitch, centerX, centerY, rotation]")
    
    if distortedPointGrid.ndim == 2:
        xe = distortedPointGrid[:,0]
        ye = distortedPointGrid[:,1]
    else:
        xe = distortedPointGrid[:,:,0]
        ye = distortedPointGrid[:,:,1]
    
    if withDistortion:
        coeffs = minimize(meritFitGridDistortion, x0, args = (xe,ye), method='Nelder-Mead')
    else:
        coeffs = minimize(meritFitGrid, x0, args = (xe,ye), method='Nelder-Mead')
    
    return coeffs.x, coeffs