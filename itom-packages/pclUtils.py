import itom
import numpy as np


def createPointCloudNormal(dataArrayNx6):

    a = dataArrayNx6
    cloud = itom.pointCloud(itom.point.PointXYZNormal)

    if a.shape[1] != 6:
        raise RuntimeError("dataArrayNx6 must have 6 columns")

    for i in range(0, a.shape[0]):
        n = a[i, 3:6]
        p = itom.point(
            itom.point.PointXYZNormal,
            xyz=a[i, 0:3],
            normal=n / np.linalg.norm(n),
            curvature=0,
        )
        cloud.append(p)

    return cloud
