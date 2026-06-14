"""Mesh
=======

"""

import numpy as np
from itom import pointCloud
from itom import algorithms
from itom import dataObject
from itom import polygonMesh
# sphinx_gallery_thumbnail_path = '11_demos/_static/_thumb/demoMesh.png'


[X, Y] = np.meshgrid(range(0, 2), range(0, 2))
Z = np.array([[0, 0], [1, 1]])

cloud = pointCloud.fromXYZ(
    dataObject(X.astype("float32")),
    dataObject(Y.astype("float32")),
    dataObject(Z.astype("float32")),
)
indices = np.array([[0, 1, 3, 2]])

mesh = polygonMesh.fromCloudAndPolygons(cloud, indices)
meshOut = polygonMesh()
algorithms.pclMeshTriangulation(mesh, meshOut)
