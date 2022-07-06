"""Cloud and mesh visualization
===============================

"""

import numpy as np
from itom import dataObject
from itom import plot
from itom import polygonMesh
# sphinx_gallery_thumbnail_path = '11_demos/_static/_thumb/demoCloudAndMeshVisualization.png'

try:
    from itom import pointCloud
except Exception as ex:
    ui.msgInformation(
        "PointCloud missing",
        "your itom version is compiled without support of pointClouds",
    )
    raise ex


###############################################################################
# Create a ``dataObject`` with X, Y and Z values of a topography
# as well as a 2.5D topography in terms of a ``dataObject``.
[X, Y] = np.meshgrid(np.arange(0, 100, 0.25), np.arange(0, 100, 0.25))
zValues = np.sin(X * 2) + np.cos(Y * 0.5)
# further intensity
intensity = np.random.rand(*X.shape)
# further color information
colorValues = dataObject.randN([X.shape[0], X.shape[1]], "rgba32")
topography = dataObject(zValues).astype("float32")
topography.axisScales = (0.1, 0.1)
topography[0, 0] = float("nan")

mesh_quads = polygonMesh.fromTopography(topography)
mesh_triangles = polygonMesh.fromTopography(topography, triangulationType=1)

[i, h] = plot(mesh_quads, "vtk3dvisualizer")
h.call("addMesh", mesh_triangles, "mesh_triangles")

###############################################################################
# .. image:: ../../_static/demoColoredShapes_1.png
#    :width: 100%
