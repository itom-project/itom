import numpy as np
from itom import *


def demo_pcl():
    # create a new point cloud
    pcl = pointCloud(point.PointXYZ)

    # add 8 points of a unit cube
    pcl.append(point(point.PointXYZ, (0, 0, 0)))
    pcl.append(point(point.PointXYZ, (1, 0, 0)))
    pcl.append(point(point.PointXYZ, (0, 1, 0)))
    pcl.append(point(point.PointXYZ, (1, 1, 0)))
    pcl.append(point(point.PointXYZ, (0, 0, 1)))
    pcl.append(point(point.PointXYZ, (1, 0, 1)))
    pcl.append(point(point.PointXYZ, (0, 1, 1)))
    pcl.append(point(point.PointXYZ, (1, 1, 1)))

    # polygons:
    polygons = np.array([[0, 1, 2], [0, 1, 3]])

    mesh = polygonMesh.fromCloudAndPolygons(pcl, polygons)

    # polygonMesh(pcl, vertices)


if __name__ == "__main__":
    demo_pcl()
