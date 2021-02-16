import numpy as np
from itom import *


def demo_pcl_mesh():
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
    filter("pclMeshTriangulation", mesh, meshOut)


if __name__ == "__main__":
    demo_pcl_mesh()
