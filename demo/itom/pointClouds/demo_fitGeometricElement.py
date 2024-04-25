"""Fit geometric element
========================

Fit geometric elements to pointClouds.
"""

import numpy as np
import math as mathe
from itom import pointCloud
from itom import dataObject
from itom import algorithms
from itom import plot
# sphinx_gallery_thumbnail_path = '11_demos/_static/_thumb/demoFitGeometricElements.png'

geometryList = [
    "circle2D",
    "circle3D",
    "sphere",
    "cylinder",
    "line",
    "plane",
]
cloud = pointCloud()

for fitGeometry in geometryList:
    if fitGeometry == "line":
        X = np.arange(-3.1, 3.1, 0.1) * 3
        Y = np.arange(-3.1, 3.1, 0.1) * 3
        Z = np.ones(X.shape, X.dtype)
        cloud = pointCloud.fromXYZ(
            dataObject(X.astype("float32")),
            dataObject(Y.astype("float32")),
            dataObject(Z.astype("float32")),
        )
        # pclNorm = pointCloud()
        # filter("pclEstimateNormals", cloud, pclNorm)
        # [cPt, cAxis, cInl] = filter("pclFitLine", pclNorm, optimizeParameters=0)

        [cPt, cAxis, cInl] = algorithms.pclFitLine(cloud, optimizeParameters=0)
        print(
            "The line is at ({}, {}, {}) with the direction ({}, {}, {})".format(
                cPt[0], cPt[1], cPt[2], cAxis[0], cAxis[1], cAxis[2]
            )
        )

    elif fitGeometry == "plane":
        [X, Y] = np.meshgrid(np.arange(-2.0, 2.0, 0.1), np.arange(-2.0, 2.0, 0.1))
        Z = np.ones(X.shape, X.dtype) + Y * np.sin(45 * np.pi / 180)
        Y *= np.cos(45 * np.pi / 180)
        cloud = pointCloud.fromXYZ(
            dataObject(X.astype("float32")),
            dataObject(Y.astype("float32")),
            dataObject(Z.astype("float32")),
        )

        [cVec, cPt, cInl] = algorithms.pclFitPlane(cloud, 1, optimizeParameters=0)

        print(
            f"The plane's direction is ({cVec[0]}, {cVec[1]}, {cVec[2]}) with the constant {cPt}"
        )

    elif fitGeometry == "circle2D":
        X = np.cos(np.arange(-3.1, 3.1, 0.1)) * 3
        Y = np.sin(np.arange(-3.1, 3.1, 0.1)) * 3
        Z = np.ones(X.shape, X.dtype)
        cloud = pointCloud.fromXYZ(
            dataObject(X.astype("float32")),
            dataObject(Y.astype("float32")),
            dataObject(Z.astype("float32")),
        )

        [cPt, cRad, cInl] = algorithms.pclFitCircle2D(
            cloud, [1, 6], optimizeParameters=0
        )

        print(f"The circle has a radius {cRad} and is centered at ({cPt[0]}, {cPt[1]})")

    elif fitGeometry == "circle3D":
        X = np.cos(np.arange(-3.1, 3.1, 0.1)) * 3
        Y = np.sin(np.arange(-3.1, 3.1, 0.1)) * 3
        Z = np.ones(X.shape, X.dtype) + Y * np.sin(45 * np.pi / 180)
        Y *= np.cos(45 * np.pi / 180)
        cloud = pointCloud.fromXYZ(
            dataObject(X.astype("float32")),
            dataObject(Y.astype("float32")),
            dataObject(Z.astype("float32")),
        )

        [cPt, cNormal, cRad, cInl] = algorithms.pclFitCircle3D(
            cloud, [1, 6], optimizeParameters=0
        )

        angle = (
            mathe.acos(
                cNormal[2]
                / (
                    cNormal[0] * cNormal[0]
                    + cNormal[1] * cNormal[1]
                    + cNormal[2] * cNormal[2]
                )
            )
            * 180
            / np.pi
        )

        angle = np.mod(angle, 90)

        print(
            "The circle has a radius {} and a angle of {} and is centered at ({}, {}, {})".format(
                cRad, angle, cPt[0], cPt[1], cPt[2]
            )
        )

    elif fitGeometry == "sphere":
        [X, Y] = np.meshgrid(np.arange(-2.0, 2.0, 0.1), np.arange(-2.0, 2.0, 0.1))
        Z = np.sqrt(9 - Y * Y - X * X)
        cloud = pointCloud.fromXYZ(
            dataObject(X.astype("float32")),
            dataObject(Y.astype("float32")),
            dataObject(Z.astype("float32")),
        )

        [cPt, cRad, cInl] = algorithms.pclFitSphere(cloud, [1, 6], optimizeParameters=0)
        print(
            f"The sphere has a radius {cRad} and is centered at ({cPt[0]}, {cPt[1]}, {cPt[2]})"
        )
    elif fitGeometry == "cylinder":
        [X, Y] = np.meshgrid(np.arange(-2.0, 2.0, 0.1), np.arange(-2.0, 2.0, 0.1))
        Z = np.sqrt(9 - Y * Y)
        cloud = pointCloud.fromXYZ(
            dataObject(X.astype("float32")),
            dataObject(Y.astype("float32")),
            dataObject(Z.astype("float32")),
        )

        # For cylinder fits we need normals defined
        pclNorm = pointCloud()
        algorithms.pclEstimateNormals(cloud, pclNorm)

        [cPt, cAxis, cRad, cInl] = algorithms.pclFitCylinder(
            pclNorm, [1, 6], optimizeParameters=0
        )
        print(
            "The cylinder has a radius {} and its axis is at ({}, {}, {}) with the direction ({}, {}, {})".format(
                cRad, cPt[0], cPt[1], cPt[2], cAxis[0], cAxis[1], cAxis[2]
            )
        )
    elif fitGeometry == "cone":
        # Not defined yet
        [X, Y] = np.meshgrid(np.arange(-2.0, 2.0, 0.1), np.arange(-2.0, 2.0, 0.1))
        Z = np.sqrt(Y * Y + X * X)
        cloud = pointCloud.fromXYZ(
            dataObject(X.astype("float32")),
            dataObject(Y.astype("float32")),
            dataObject(Z.astype("float32")),
        )

    plot(cloud)
