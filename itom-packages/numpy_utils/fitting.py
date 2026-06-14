import numpy as np


def lineFit3D(pointArray):
    """
    Calculates the best fit line into a given set of points (3D).
    This line would minimize the squared distance between the real
    point and its projection.

    This method uses for this the principal component analysis.
    One way to define it is the line whose direction vector is the eigenvector of
    the covariance matrix corresponding to the largest eigenvalue, that passes
    through the mean of your data. That said, eig(cov(data)) is a really bad way to
    calculate it, since it does a lot of needless computation and copying and is
    potentially less accurate than using svd.

    Input
    -----
    - 3xN pointArray (numpy array)

    Output
    ------
    - centre of gravity (mean value) of the point cloud [3x1]
    - fitted unit vector [3x1]

    http://stackoverflow.com/questions/2298390/fitting-a-line-in-3d
    """

    pointArray = np.array(pointArray)
    [m, n] = pointArray.shape

    if m != 3:
        raise RuntimeError("array must have shape 3xN")

    if n < 2:
        raise RuntimeError("you must indicate at least two different points")

    # Calculate the mean of the points, i.e. the 'center' of the cloud
    datamean = pointArray.mean(axis=1)
    datamean = datamean[:, np.newaxis]

    # Do an SVD on the mean-centered data.
    uu, dd, vv = np.linalg.svd((pointArray - datamean).T)

    # Now vv[0] contains the first principal component, i.e. the direction
    # vector of the 'best fit' line in the least squares sense.
    return (datamean.T, vv[0:1, :])


if __name__ == "__main__":

    # Generate some data that lies along a line
    x = np.mgrid[-2:5:120j]
    y = np.mgrid[1:9:120j]
    z = np.mgrid[-5:3:120j]

    data = np.concatenate(
        (x[:, np.newaxis], y[:, np.newaxis], z[:, np.newaxis]), axis=1
    )

    # Perturb with some Gaussian noise
    data += np.random.normal(size=data.shape) * 0.4

    [datamean, vv] = lineFit3D(data.T)

    # I use -7, 7 since the spread of the data is roughly 14
    # and we want it to have mean 0 (like the points we did
    # the svd on). Also, it's a straight line, so we only need 2 points.
    linepts = vv * np.mgrid[-7:7:2j][:, np.newaxis]

    # shift by the mean to get the line in the right place
    linepts += datamean

    p1 = datamean
    p2 = p1 + vv
    p3 = np.vstack((p1, p2))

    # Verify that everything looks right.
    import matplotlib

    matplotlib.use("module://mpl_itom.backend_itomagg", False)
    import matplotlib.pyplot as plt
    import mpl_toolkits.mplot3d as m3d

    ax = m3d.Axes3D(plt.figure())
    ax.scatter3D(*data.T)
    ax.plot3D(*linepts.T)
    ax.plot3D(*p3.T)
    plt.show()
