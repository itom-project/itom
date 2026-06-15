"""Nearest neighbors
=================

"""

import numpy as np
from scipy import spatial
# sphinx_gallery_thumbnail_path = '11_demos/_static/_thumb/demoNearestNeighbors.png'

###############################################################################
# Make a grid of 0:9 in X and 0:20 in Y.
[X, Y] = np.meshgrid(range(0, 10), range(0, 20))

###############################################################################
#  Make one XY array [10*20 x 2] where each line is the x, y position of one point.
XY = np.vstack([X.flatten(), Y.flatten()]).transpose()

###############################################################################
# Make the fast search tree.
kdtree = spatial.cKDTree(XY, leafsize=4)

###############################################################################
# Choose some random points in the grid and get the 8 nearest neighbours.
rand = np.random.randint(0, XY.shape[1], size=(20,))

for r in rand:
    # random point:
    point = XY[r, :]
    # query the 8 nearest neightbours with an euclidian distance
    [dists, indices] = kdtree.query(point, k=8, p=2)
    print("Nearest points to point:", point)
    print("------------------------------------------")
    for idx in range(0, len(indices)):
        print("    ", idx, ".:", XY[indices[idx], :], " -> dist:", dists[idx])
