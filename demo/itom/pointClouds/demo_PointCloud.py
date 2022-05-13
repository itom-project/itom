"""PointCloud 
=============

This demo shows how you can work with
pointClouds based on the ``pointCloudLibrary`` in ``itom``."""


import numpy as np
from itom import pointCloud
from itom import polygonMesh
from itom import point
# sphinx_gallery_thumbnail_path = '11_demos/_static/_thumb/demoPCL.png'

###############################################################################
# Create a new ``pointCloud``.
pcl = pointCloud(point.PointXYZ)

###############################################################################
# Add 8 points of a unit cube.
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