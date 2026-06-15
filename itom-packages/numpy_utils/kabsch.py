import numpy as np
from numpy import linalg as linalg


def kabsch_algorithm(P, Q, m=None):
    """
    [U, r, lrms] = kabsch_algorithm(P, Q, m)

    Find the Least Root Mean Square distance
    between two sets of N points in D dimensions
    and the rigid transformation (i.e. translation and rotation)
     to employ in order to bring one set that close to the other,
     Using the Kabsch (1976) algorithm.
     Note that the points are paired, i.e. we know which point in one set
     should be compared to a given point in the other set.

     References:
     1) Kabsch W. A solution for the best rotation to relate two sets of vectors. Acta Cryst A 1976;32:9223.
     2) Kabsch W. A discussion of the solution for the best rotation to relate two sets of vectors. Acta Cryst A 1978;34:8278.
     3) http://cnx.org/content/m11608/latest/
     4) http://en.wikipedia.org/wiki/Kabsch_algorithm

     We slightly generalize, allowing weights given to the points.
     Those weights are determined a priori and do not depend on the distances.

     We work in the convention that points are column vectors;
     some use the convention where they are row vectors instead.

     Input  variables:
      P : a D*N matrix where P(a,i) is the a-th coordinate of the i-th point
          in the 1st representation
      Q : a D*N matrix where Q(a,i) is the a-th coordinate of the i-th point
          in the 2nd representation
      m : (Optional) a row vector of length N giving the weights, i.e. m(i) is
          the weight to be assigned to the deviation of the i-th point.
          If not supplied, we take by default the unweighted (or equal weighted)
          m(i) = 1/N.
          The weights do not have to be normalized;
          we divide by the sum to ensure sum_{i=1}^N m(i) = 1.
          The weights must be non-negative with at least one positive entry.

     Output variables:
      U : a proper orthogonal D*D matrix, representing the rotation
      r : a D-dimensional column vector, representing the translation
      lrms: the Least Root Mean Square

     Definition:
      U is the rotation matrix, such that Q = U * P + r (approx.)

     Details:
      If p_i, q_i are the i-th point (as a D-dimensional column vector)
      in the two representations, i.e. p_i = P(:,i) etc., and for
      p_i' = U p_i + r      (' does not stand for transpose!)
      we have p_i' ~ q_i, that is,
      lrms = sqrt(sum_{i=1}^N m(i) (p_i' - q_i)^2)
      is the minimal rms when going over the possible U and r.
      (assuming the weights are already normalized).
    """

    sz1 = P.shape
    sz2 = Q.shape

    if not isinstance(P, np.matrix) or (not isinstance(Q, np.matrix)):
        P = np.matrix(P)
        Q = np.matrix(Q)
        if not isinstance(P, np.matrix) or (not isinstance(Q, np.matrix)):
            raise RuntimeError("P and Q must be np.matrix")

    if len(sz1) != 2 or len(sz2) != 2:
        raise RuntimeError("P and Q must be matrices")

    if np.any(sz1 != sz2):
        raise RuntimeError("P and Q must be of same size")

    D = sz1[0]  # dimension of space
    N = sz1[1]  # number of points

    if m is None:
        m = np.matrix(np.ones([1, N])) / N  # m not supplied - use default
    else:
        m = np.matrix(m)
        if m.shape != (1, N):
            raise RuntimeError("m must be a row vector of length N")

        if True in (m < 0):
            raise RuntimeError("m must have non-negative entries")

        if np.sum(m) == 0:
            raise RuntimeError("m must contain some positive entry")

        m = m / np.sum(m)  # normalize so that weights sum to 1

    p0 = P * m.T  # the centroid of P
    q0 = Q * m.T  # the centroid of Q
    v1 = np.matrix(np.ones([1, N]))  # row vector of N ones
    P = P - p0 * v1  # translating P to center the origin
    Q = Q - q0 * v1  # translating Q to center the origin

    # C is a covariance matrix of the coordinates
    # C = P * np.matrix( np.diag(m) ) * Q.T
    # but this is inefficient, involving an N*N matrix, while typically D << N.
    # so we use another way to compute Pdm = P*diag(m)
    Pdm = np.matrix(np.zeros([D, N]))
    for i in range(0, N):
        Pdm[:, i] = m[0, i] * P[:, i]

    C = Pdm * Q.T
    # C = P*Q' / N ;       % (for the non-weighted case)
    [V, S, W] = linalg.svd(C)  # singular value decomposition
    W = W.T
    S = np.matrix(np.diag(S))

    I = np.eye(D)

    if linalg.det(V * W.T) < 0:  # more numerically stable than using (det(C) < 0)
        I[D - 1, D - 1] = -1

    U = W * I * V.T
    r = q0 - U * p0

    Diff = U * P - Q  # P, Q already centered
    # lrms = sqrt(sum(sum(Diff.*Diff))/N) ; % (for the non-weighted case
    lrms = 0
    for i in range(0, N):
        lrms = lrms + m[0, i] * Diff[:, i].T * Diff[:, i]

    lrms = np.sqrt(lrms)

    return [U, r, lrms]


############################################################################
def umeyama_algorithm(P, Q):
    """
    [U, r, s, lrms] = umeyama_algorithm(P, Q, m)

    Find the Least Root Mean Square distance
    between two sets of N points in D dimensions
    and the rigid transformation (i.e. translation and rotation)
     to employ in order to bring one set that close to the other,
     Using the Umeyama (1991) algorithm. -> http://web.stanford.edu/class/cs273/refs/umeyama.pdf
     Note that the points are paired, i.e. we know which point in one set
     should be compared to a given point in the other set.

     References:
     1) http://web.stanford.edu/class/cs273/refs/umeyama.pdf

     We work in the convention that points are column vectors;
     some use the convention where they are row vectors instead.

     Input  variables:
      P : a D*N matrix where P(a,i) is the a-th coordinate of the i-th point
          in the 1st representation
      Q : a D*N matrix where Q(a,i) is the a-th coordinate of the i-th point
          in the 2nd representation

     Output variables:
      U : a proper orthogonal D*D matrix, representing the rotation
      r : a D-dimensional column vector, representing the translation
      s : a scaling factor (the same for all dimensions)
      lrms: the Least Root Mean Square

     Definition:
      U is the rotation matrix, such that Q = s * U * P + r (approx.)
    """

    sz1 = P.shape
    sz2 = Q.shape

    if not isinstance(P, np.matrix) or (not isinstance(Q, np.matrix)):
        P = np.matrix(P)
        Q = np.matrix(Q)
        if not isinstance(P, np.matrix) or (not isinstance(Q, np.matrix)):
            raise RuntimeError("P and Q must be np.matrix")

    if len(sz1) != 2 or len(sz2) != 2:
        raise RuntimeError("P and Q must be matrices")

    if np.any(sz1 != sz2):
        raise RuntimeError("P and Q must be of same size")

    D = sz1[0]  # dimension of space
    N = sz1[1]  # number of points
    m = np.matrix(np.ones([1, N])) / N  # m not supplied - use default

    p0 = P * m.T  # the centroid of P
    q0 = Q * m.T  # the centroid of Q (target)
    v1 = np.matrix(np.ones([1, N]))  # row vector of N ones
    P_centered = P - p0 * v1  # translating P to center the origin
    Q_centered = Q - q0 * v1  # translating Q to center the origin

    # C is a covariance matrix of the coordinates
    # C = P * np.matrix( np.diag(m) ) * Q.T
    # but this is inefficient, involving an N*N matrix, while typically D << N.
    # so we use another way to compute Pdm = P*diag(m)
    Pdm = np.matrix(np.zeros([D, N]))
    f_sd2_q = 0
    f_sd2_p = 0

    for i in range(0, N):
        f_sd2_q += linalg.norm(Q_centered[:, i]) ** 2
        f_sd2_p += linalg.norm(P_centered[:, i]) ** 2
        Pdm[:, i] = m[0, i] * P_centered[:, i]

    C = Pdm * Q_centered.T
    # C = P*Q' / N ;       % (for the non-weighted case)
    [V, S, W] = linalg.svd(C)  # singular value decomposition, S are singular values
    W = W.T
    S = np.matrix(np.diag(S))

    I = np.eye(D)

    if linalg.det(V * W.T) < 0:  # more numerically stable than using (det(C) < 0)
        I[D - 1, D - 1] = -1

    f_scale = np.sum(S * I) / f_sd2_q
    f_inv_scale = np.sum(S * I) / f_sd2_p  # only one of those is needed
    scale = f_inv_scale * N

    U = W * I * V.T
    r = q0 - scale * U * p0
    ######################

    Diff = scale * U * P_centered - Q_centered  # P, Q already centered
    # lrms = sqrt(sum(sum(Diff.*Diff))/N) ; % (for the non-weighted case
    lrms = 0
    for i in range(0, N):
        lrms = lrms + m[0, i] * Diff[:, i].T * Diff[:, i]

    lrms = np.sqrt(lrms)

    return [U, r, scale, lrms]


if __name__ == "__main__":

    A = np.array([[1, 1, 3], [3, 2, 1], [4, 5, 6]])
    B = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
    [U, r, lrms] = kabsch_algorithm(A, B)
    print("U:", U, "\nr:", r, "\nlrms:", lrms)

    A2 = np.array([[1, 1, 3], [3, 2, 1], [4, 5, 6], [7, -1, 2]]).T
    dx = 0.2
    dy = 1
    dz = 2
    B2 = np.array(
        [
            [1 + dx, 1 + dy, 3 + dz],
            [3 + dx, 2 + dy, 1 + dz],
            [4 + dx, 5 + dy, 6 + dz],
            [7 + dx, -1 + dy, 2 + dz],
        ]
    ).T
    [U1, r1, lrms1] = kabsch_algorithm(A2, B2)
    print("U:", U1, "\nr:", r1, "\nlrms:", lrms1)

    [U2, r2, s2, lrms2] = umeyama_algorithm(A2, B2)

    A3 = np.eye(3)
    B3 = np.eye(3)
    [U, r, lrms] = kabsch_algorithm(A3, B3)
    print("U:", U, "\nr:", r, "\nlrms:", lrms)

    A4 = np.array([[1, 1, 3], [3, 2, 1], [4, 5, 6]])
    B4 = np.array([[2, -1, -3], [-3, 5, 2], [10, 11, 14]])
    [U, r, lrms] = kabsch_algorithm(A4, B4)
    print("U:", U, "\nr:", r, "\nlrms:", lrms)

    # Benchmark with other library for affine transformation
    # from numpy_utils.transformations import transformations as trafo
    # print("U2:",trafo.affine_matrix_from_points(A,B,False,False,True))
