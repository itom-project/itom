"""Cloud
========

"""

import numpy as np
from itom import plot

try:
    from itom import pointCloud
except Exception as ex:
    ui.msgInformation(
        "PointCloud missing",
        "your itom version is compiled without support of pointClouds",
    )
    raise ex
# sphinx_gallery_thumbnail_path = '11_demos/_static/_thumb/demoCloud.png'

###############################################################################
# Create a sphere
polarRad = np.radians(np.arange(0, 360, 1))
azimuthRad = np.radians(np.arange(0, 180, 1))
nominalRadius = 5
polars, azimuths = np.meshgrid(azimuthRad, polarRad)

X = nominalRadius * np.cos(polars) * np.cos(azimuths)
Y = nominalRadius * np.sin(polars) * np.cos(azimuths)
Z = nominalRadius * np.sin(azimuths)

###############################################################################
# Flatten all X,Y,Z coordinates
X = X.flatten()
Y = Y.flatten()
Z = Z.flatten()

###############################################################################
# Create random noise in X, Y and Z direction
level = 0.3
Xnoise = (np.random.rand(len(X)) - 0.5) * level
Ynoise = (np.random.rand(len(Y)) - 0.5) * level
Znoise = (np.random.rand(len(Z)) - 0.5) * level
dist = np.sqrt(Xnoise ** 2 + Ynoise ** 2 + Znoise ** 2)

###############################################################################
# Cloud 1: perfect sphere, no intensity values
cloud1 = pointCloud.fromXYZ(X, Y, Z)

###############################################################################
# Cloud 2: noisy sphere, noise deviation as intensity, shift it a little bit in X direction
Xshift = nominalRadius * 2.2
cloud2 = pointCloud.fromXYZI(
    X + Xnoise + Xshift, Y + Ynoise, Z + Znoise, dist
)

# Plot the first cloud --> this cloud has the default name 'source_cloud_normal'
index, handle = plot(cloud1, "vtk3dvisualizer")

# parametrize cloud1
handle.call("setItemProperty", "source_cloud_normal", "ColorMode", "Z")
handle.call("setItemProperty", "source_cloud_normal", "ColorMap", "hsv")

# plot the second sphere and shift it a little bit
handle.call(
    "addPointCloud", cloud2, "cloud2"
)  # visualize cloud2 under the name 'cloud2'
handle.call(
    "setItemProperty", "cloud2", "PointSize", 2
)  # change the property PointSize of this point
handle.call("setItemProperty", "cloud2", "ColorMode", "Intensity")
handle.call("setItemProperty", "cloud2", "ColorMap", "blue2red")
handle.call("setItemProperty", "cloud2", "ColorValueRange", (0.1, 0.2))

###############################################################################
# .. image:: ../../_static/demoClouds_1.png
#    :width: 100%