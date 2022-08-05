"""PointCloud 
=============

This demo is a very short introduction to basic point clouds
and a polygon mesh, that consists of two triangles, whose corner
points are given by some points of a cloud."""


import numpy as np
from itom import pointCloud
from itom import polygonMesh
from itom import point

# sphinx_gallery_thumbnail_path = '11_demos/_static/_thumb/demoPCL.png'

###############################################################################
# Create a new, empty ``pointCloud`` of type x,y,z only.
# Point clouds can either be organized or unorganized. Organized point
# clouds have a height > 1, such that all points are organized in a 2d grid
# and the direct neighbours of every point are therefore defined.
# In unorganized point clouds, the height is always <= 1 and the size is
# equal to the width of the cloud. No point has any neighbourhood relationship
# to other points. This is usually the default.
pcl = pointCloud(point.PointXYZ)

print(
    "Initial point cloud pcl, shape (h, w): %i x %i, fields: %s, organized: %i"
    % (pcl.height, pcl.width, str(pcl.fields), pcl.organized)
)

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

print(
    "Modified point cloud pcl, shape (h, w): %i x %i, fields: %s, organized: %i"
    % (pcl.height, pcl.width, str(pcl.fields), pcl.organized)
)

for idx in range(pcl.size):
    print("Point %i: %s" % (idx + 1, str(pcl[idx])))

# A polygon mesh can be constructed from a point cloud, that indicates
# all corner points of the mesh and a table of vertices. Every entry in the
# vertices table must contain 3 (default) or 4 values and indicate the indices
# to the points in the cloud that define a triangle or quadrangle of the
# mesh. If the normal vector to such a triangle points towards the outside
# of the object, the order of the points is given by the right-hand rule, where
# the thumbs shows in the direction of the normale vector.

# create an array with two triangles. The first triangle has the corner
# points with the indices 0, 1 and 2 of the cloud. The 2nd triangle consists
# of the corner points 0, 1 and 3.
vertices = np.array([[0, 1, 2], [0, 1, 3]])

# create a mesh from the cloud and the vertices
mesh = polygonMesh.fromCloudAndPolygons(pcl, vertices)

print("Mesh of %i polygons" % mesh.nrOfPolygons)
