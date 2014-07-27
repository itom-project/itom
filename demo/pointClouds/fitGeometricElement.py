'''
demo file to fit geomtric elements to pointClouds
by Wolfram Lyda
Licenced under LGPL
Copyright twip optical solutions
'''

import numpy as np
import math as mathe

geometryList = ["circle2D", "circle3D", "sphere", "cylinder"]

fitGeometry = geometryList[0]

cloud = pointCloud()
doPlot = False

if fitGeometry == "circle2D":
    
    X = np.cos(np.arange(-3.1,3.1, 0.1)) * 3
    Y = np.sin(np.arange(-3.1,3.1, 0.1)) * 3
    Z = np.ones(X.shape, X.dtype)
    cloud = pointCloud.fromXYZ(dataObject(X.astype('float32')),dataObject(Y.astype('float32')),dataObject(Z.astype('float32')))
    
    [cPt, cRad, cInl] = filter("pclFitCircle2D", cloud,  [1, 6], optimizeParameters=1)
    
    print("The circle has a radiues {} and is centered at ({}, {})".format(cRad, cPt[0], cPt[1]))
    
elif fitGeometry == "circle3D":
    
    X = np.cos(np.arange(-3.1,3.1, 0.1)) * 3
    Y = np.sin(np.arange(-3.1,3.1, 0.1)) * 3
    Z = np.ones(X.shape, X.dtype) + Y * np.sin(45 * np.pi / 180)
    Y *= np.cos(45 * np.pi / 180)
    cloud = pointCloud.fromXYZ(dataObject(X.astype('float32')),dataObject(Y.astype('float32')),dataObject(Z.astype('float32')))
    
    [cPt, cNormal, cRad, cInl] = filter("pclFitCircle3D", cloud,  [1, 6], optimizeParameters=1)
    
    angle = mathe.acos(cNormal[2] / (cNormal[0] * cNormal[0] + cNormal[1] * cNormal[1] + cNormal[2] * cNormal[2])) * 180 / np.pi
    
    if angle > 90:
        angle = angle + 90
    elif angle < -90:
        angle = angle + 90
    
    print("The circle has a radiues {} and a angle of {} and is centered at ({}, {}, {})".format(cRad, angle, cPt[0], cPt[1], cPt[2]))
    
elif fitGeometry == "sphere":
    
    [X,Y] = np.meshgrid(np.arange(-2.0,2.0, 0.1),np.arange(-2.0,2.0, 0.1))
    Z = np.sqrt(9 - Y*Y - X*X)
    cloud = pointCloud.fromXYZ(dataObject(X.astype('float32')),dataObject(Y.astype('float32')),dataObject(Z.astype('float32')))
    
    [cPt, cRad, cInl] = filter("pclFitSphere", cloud,  [1, 6], optimizeParameters=1)
    
elif fitGeometry == "cylinder":
    
    [X,Y] = np.meshgrid(np.arange(-2.0,2.0, 0.1),np.arange(-2.0,2.0, 0.1))
    Z = np.sqrt(9 - Y*Y)
    cloud = pointCloud.fromXYZ(dataObject(X.astype('float32')),dataObject(Y.astype('float32')),dataObject(Z.astype('float32')))
    
    # For cylinder fits we need normals defined
    pclNorm = pointCloud()
    filter("pclEstimateNormals", cloud, pclNorm)
    
    [cPt, cAxis, cRad, cInl] = filter("pclFitCylinder", pclNorm,  [1, 6], optimizeParameters=1)
    
elif fitGeometry == "cone":
    
    [X,Y] = np.meshgrid(np.arange(-2.0,2.0, 0.1),np.arange(-2.0,2.0, 0.1))
    Z = np.sqrt(Y*Y + X*X)
    cloud = pointCloud.fromXYZ(dataObject(X.astype('float32')),dataObject(Y.astype('float32')),dataObject(Z.astype('float32')))
    
else:
    print ("type not defiend")
    raise

if doPlot:
    plot(cloud)

