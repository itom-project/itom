"""Cloud viewer
===============

"""

import numpy as np
from itom import pointCloud
from itom import polygonMesh
from itom import ui
from itom import dataObject

try:
    from itom import pointCloud
except Exception as ex:
    ui.msgInformation(
        "PointCloud missing",
        "your itom version is compiled without support of pointClouds",
    )
    raise ex
# sphinx_gallery_thumbnail_path = '11_demos/_static/_thumb/demoCloudAndMeshVisualization.png'


###############################################################################
# Create a data objects with X, Y and Z values of a topography
# as well as a 2.5D topography in terms of a data object.
[X, Y] = np.meshgrid(np.arange(0, 5, 0.1), np.arange(0, 5, 0.1))
Z = np.sin(X * 2) + np.cos(Y * 0.5)
I = np.random.rand(*X.shape)  # further intensity
C = dataObject.randN([X.shape[0], X.shape[1]], "rgba32")  # further color information
topography = dataObject(Z).astype("float32")
topography.axisScales = (0.1, 0.1)
topography[0, 0] = float("nan")

###############################################################################
# Create a point cloud from the X, Y and Z arrays with further intensity information.
cloud1 = pointCloud.fromXYZI(X, Y, Z, I)

###############################################################################
# Create a point cloud from the topography image with further colour information.
cloud2 = pointCloud.fromTopography(topography, color=C)

###############################################################################
# Create a point cloud from the X, Y and Z arrays with further colour information.
cloud3 = pointCloud.fromXYZRGBA(X, Y, Z, C)

###############################################################################
# Create a point cloud from the X, Y and Z arrays with the Z-values as intensity information.
cloud4 = pointCloud.fromXYZI(X, Y, Z - 0.1, Z)

###############################################################################
# Manually create triangular polygons for the whole surface
# the polygons are regularly distributed and each rectangle is divided into two polygons.
polygons = dataObject.zeros([2 * 49 * 49, 3], "uint16")
c = 0
for row in range(0, 49):
    for col in range(0, 49):
        polygons[c, 0] = row * 50 + col
        polygons[c, 1] = (row + 1) * 50 + col
        polygons[c, 2] = row * 50 + 1 + col
        c += 1
for row in range(0, 49):
    for col in range(0, 49):
        polygons[c, 0] = (row + 1) * 50 + col
        polygons[c, 1] = (row + 1) * 50 + col + 1
        polygons[c, 2] = row * 50 + col + 1
        c += 1

###############################################################################
# Create polygonal mesh structure from cloud3 and polygons.
mesh = polygonMesh.fromCloudAndPolygons(cloud3, polygons)

###############################################################################
# As alternative approach you can directly create the same polygonal mesh
# from the point cloud if you know that the point cloud is organized, hence,
# the points are located like in a regular grid.
mesh2 = polygonMesh.fromOrganizedCloud(cloud2)

###############################################################################
# Create GUI (3D Viewer)
gui = ui("cloudViewer.ui", ui.TYPEWINDOW)

# gui.plot.call("addPointCloud",cloud1,"cloud1")
# gui.plot.call("addPointCloud",cloud2,"cloud2")
# gui.plot.call("addPointCloud",cloud3,"cloud3")
gui.plot.call(
    "addPointCloud", cloud4, "cloud4"
)  # visualize cloud4 under the name 'cloud4'
gui.plot.call(
    "setItemProperty", "cloud4", "PointSize", 10
)  # change the property PointSize of this point
# gui.plot.call("addMesh",mesh,"mesh")
gui.plot.call("addMesh", mesh2, "mesh2")  # visualize the mesh2 under the name 'mesh2'
gui.show()
