""" Linear Algebra Helper Routines """

__docformat__ = "restructuredtext en"

from warnings import warn
import numpy
import scipy
import scipy.sparse as sparse

# from scipy.linalg.lapack import get_lapack_funcs
# from scipy.linalg import calc_lwork

__all__ = [
    "approximate_spectral_radius",
    "infinity_norm",
    "norm",
    "residual_norm",
    "condest",
    "cond",
    "issymm",
    "pinv_array",
]


def norm(x, pnorm="2"):
    """
    2-norm of a vector

    Parameters
    ----------
    x : array_like
        Vector of complex or real values

    pnorm :

    Returns
    -------
    n : float
        2-norm of a vector

    Notes
    -----
    - currently 1+ order of magnitude faster than scipy.linalg.norm(x), which calls
      sqrt(numpy.sum(real((conjugate(x)*x)),axis=0)) resulting in an extra copy
    - only handles the 2-norm for vectors

    See Also
    --------
    scipy.linalg.norm : scipy general matrix or vector norm
    """

    # TODO check dimensions of x
    # TODO speedup complex case

    x = numpy.ravel(x)

    if pnorm == "2":
        return numpy.sqrt(numpy.inner(x.conj(), x).real)
    elif pnorm == "inf":
        return numpy.max(numpy.abs(x))
    else:
        raise ValueError("Only the 2-norm and infinity-norm are supported")


def infinity_norm(A):
    """
    Infinity norm of a matrix (maximum absolute row sum).

    Parameters
    ----------
    A : csr_matrix, csc_matrix, sparse, or numpy matrix
        Sparse or dense matrix

    Returns
    -------
    n : float
        Infinity norm of the matrix

    Notes
    -----
    - This serves as an upper bound on spectral radius.
    - csr and csc avoid a deep copy
    - dense calls scipy.linalg.norm

    See Also
    --------
    scipy.linalg.norm : dense matrix norms

    Examples
    --------
    >>> import numpy
    >>> from scipy.sparse import spdiags
    >>> from pyamg.util.linalg import infinity_norm
    >>> n=10
    >>> e = numpy.ones((n,1)).ravel()
    >>> data = [ -1*e, 2*e, -1*e ]
    >>> A = spdiags(data,[-1,0,1],n,n)
    >>> print infinity_norm(A)
    4.0
    """

    if sparse.isspmatrix_csr(A) or sparse.isspmatrix_csc(A):
        # avoid copying index and ptr arrays
        abs_A = A.__class__((numpy.abs(A.data), A.indices, A.indptr), shape=A.shape)
        return (abs_A * numpy.ones((A.shape[1]), dtype=A.dtype)).max()
    elif sparse.isspmatrix(A):
        return (abs(A) * numpy.ones((A.shape[1]), dtype=A.dtype)).max()
    else:
        return numpy.dot(numpy.abs(A), numpy.ones((A.shape[1],), dtype=A.dtype)).max()


def residual_norm(A, x, b):
    """Compute ||b - A*x||"""

    return norm(numpy.ravel(b) - A * numpy.ravel(x))


def axpy(x, y, a=1.0):
    """
    Quick level-1 call to blas::
    y = a*x+y

    Parameters
    ----------
    x : array_like
        nx1 real or complex vector
    y : array_like
        nx1 real or complex vector
    a : float
        real or complex scalar

    Returns
    -------
    y : array_like
        Input variable y is rewritten

    Notes
    -----
    The call to get_blas_funcs automatically determines the prefix for the blas
    call.
    """
    from scipy.lib.blas import get_blas_funcs

    fn = get_blas_funcs(["axpy"], [x, y])[0]
    fn(x, y, a)


# def approximate_spectral_radius(A, tol=0.1, maxiter=10, symmetric=False):
#    """approximate the spectral radius of a matrix
#
#    Parameters
#    ----------
#
#    A : {dense or sparse matrix}
#        E.g. csr_matrix, csc_matrix, ndarray, etc.
#    tol : {scalar}
#        Tolerance of approximation
#    maxiter : {integer}
#        Maximum number of iterations to perform
#    symmetric : {boolean}
#        True if A is symmetric, False otherwise (default)
#
#    Returns
#    -------
#        An approximation to the spectral radius of A
#
#    """
#    if symmetric:
#        method = eigen_symmetric
#    else:
#        method = eigen
#
#    return norm( method(A, k=1, tol=0.1, which='LM', maxiter=maxiter, return_eigenvectors=False) )


def _approximate_eigenvalues(A, tol, maxiter, symmetric=None):
    """Used by approximate_spectral_radius and condest"""

    from scipy.sparse.linalg import aslinearoperator

    A = aslinearoperator(A)  # A could be dense or sparse, or something weird

    ##
    # Choose tolerance for deciding if break-down has occurred
    t = A.dtype.char
    eps = numpy.finfo(numpy.float).eps
    feps = numpy.finfo(numpy.single).eps
    geps = numpy.finfo(numpy.longfloat).eps
    _array_precision = {"f": 0, "d": 1, "g": 2, "F": 0, "D": 1, "G": 2}
    breakdown = {0: feps * 1e3, 1: eps * 1e6, 2: geps * 1e6}[_array_precision[t]]

    if A.shape[0] != A.shape[1]:
        raise ValueError("expected square matrix")

    maxiter = min(A.shape[0], maxiter)

    v0 = scipy.rand(A.shape[1], 1)
    if A.dtype == complex:
        v0 = v0 + 1.0j * scipy.rand(A.shape[1], 1)

    v0 /= norm(v0)

    H = numpy.zeros((maxiter + 1, maxiter), dtype=A.dtype)
    V = [v0]

    for j in range(maxiter):
        w = A * V[-1]

        if symmetric:
            if j >= 1:
                H[j - 1, j] = beta
                w -= beta * V[-2]

            alpha = numpy.dot(numpy.conjugate(numpy.ravel(w)), numpy.ravel(V[-1]))
            H[j, j] = alpha
            w -= alpha * V[-1]  # axpy(V[-1],w,-alpha)

            beta = norm(w)
            H[j + 1, j] = beta

            if H[j + 1, j] < breakdown:
                break

            w /= beta

            V.append(w)
            V = V[-2:]  # retain only last two vectors

        else:
            # orthogonalize against Vs
            for i, v in enumerate(V):
                H[i, j] = numpy.dot(numpy.conjugate(numpy.ravel(v)), numpy.ravel(w))
                w = w - H[i, j] * v

            H[j + 1, j] = norm(w)

            if H[j + 1, j] < breakdown:
                break

            w = w / H[j + 1, j]
            V.append(w)

            # if upper 2x2 block of Hessenberg matrix H is almost symmetric,
            # and the user has not explicitly specified symmetric=False,
            # then switch to symmetric Lanczos algorithm
            # if symmetric is not False and j == 1:
            #    if abs(H[1,0] - H[0,1]) < 1e-12:
            #        #print "using symmetric mode"
            #        symmetric = True
            #        V = V[1:]
            #        H[1,0] = H[0,1]
            #        beta = H[2,1]

    # print "Approximated spectral radius in %d iterations" % (j + 1)

    from scipy.linalg import eigvals

    return eigvals(H[: j + 1, : j + 1])


def approximate_spectral_radius(A, tol=0.1, maxiter=10, symmetric=None):
    """
    Approximate the spectral radius of a matrix

    Parameters
    ----------

    A : {dense or sparse matrix}
        E.g. csr_matrix, csc_matrix, ndarray, etc.
    tol : {scalar}
        Tolerance of approximation (currently unused)
    maxiter : {integer}
        Maximum number of iterations to perform
    symmetric : {boolean}
        True  - if A is symmetric
                Lanczos iteration is used (more efficient)
        False - if A is non-symmetric (default
                Arnoldi iteration is used (less efficient)

    Returns
    -------
    An approximation to the spectral radius of A

    Notes
    -----
    The spectral radius is approximated by looking at the Ritz eigenvalues.
    Arnoldi iteration (or Lanczos) is used to project the matrix A onto a
    Krylov subspace: H = Q* A Q.  The eigenvalues of H (i.e. the Ritz
    eigenvalues) should represent the eigenvalues of A in the sense that the
    minimum and maximum values are usually well matched (for the symmetric case
    it is true since the eigenvalues are real).

    References
    ----------
    .. [1] Z. Bai, J. Demmel, J. Dongarra, A. Ruhe, and H. van der Vorst, editors.
       "Templates for the Solution of Algebraic Eigenvalue Problems: A Practical
       Guide", SIAM, Philadelphia, 2000.

    Examples
    --------
    >>> from pyamg.util.linalg import approximate_spectral_radius
    >>> import numpy
    >>> from scipy.linalg import eigvals, norm
    >>> A = numpy.array([[1,0],[0,1]])
    >>> print approximate_spectral_radius(A,maxiter=3)
    1.0
    >>> print max([norm(x) for x in eigvals(A)])
    1.0
    """

    if not hasattr(A, "rho"):
        ev = _approximate_eigenvalues(A, tol, maxiter, symmetric)
        rho = numpy.max(numpy.abs(ev))
        if sparse.isspmatrix(A):
            A.rho = rho

        return rho

    else:
        return A.rho


def condest(A, tol=0.1, maxiter=25, symmetric=False):
    """Estimates the condition number of A

    Parameters
    ----------
    A   : {dense or sparse matrix}
        e.g. array, matrix, csr_matrix, ...
    tol : {float}
        Approximation tolerance, currently not used
    maxiter: {int}
        Max number of Arnoldi/Lanczos iterations
    symmetric : {bool}
        If symmetric use the far more efficient Lanczos algorithm,
        Else use Arnoldi

    Returns
    -------
    Estimate of cond(A) with \|lambda_max\| / \|lambda_min\|
    through the use of Arnoldi or Lanczos iterations, depending on
    the symmetric flag

    Notes
    -----
    The condition number measures how large of a change in the
    the problems solution is caused by a change in problem's input.
    Large condition numbers indicate that small perturbations
    and numerical errors are magnified greatly when solving the system.

    Examples
    --------
    >>> import numpy
    >>> from pyamg.util.linalg import condest
    >>> condest(numpy.array([[1,0],[0,2]]))
    1.0

    """

    ev = _approximate_eigenvalues(A, tol, maxiter, symmetric)

    return numpy.max([norm(x) for x in ev]) / min([norm(x) for x in ev])


def cond(A):
    """Returns condition number of A

    Parameters
    ----------
    A   : {dense or sparse matrix}
        e.g. array, matrix, csr_matrix, ...

    Returns
    -------
    2-norm condition number through use of the SVD
    Use for small to moderate sized dense matrices.
    For large sparse matrices, use condest.

    Notes
    -----
    The condition number measures how large of a change in the
    the problems solution is caused by a change in problem's input.
    Large condition numbers indicate that small perturbations
    and numerical errors are magnified greatly when solving the system.

    Examples
    --------
    >>> import numpy
    >>> from pyamg.util.linalg import cond
    >>> condest(numpy.array([[1,0],[0,2]]))
    1.0

    """

    if A.shape[0] != A.shape[1]:
        raise ValueError("expected square matrix")

    if sparse.isspmatrix(A):
        A = A.todense()

    # 2-Norm Condition Number
    from scipy.linalg import svd

    U, Sigma, Vh = svd(A)
    return numpy.max(Sigma) / min(Sigma)


def issymm(A, fast_check=True, tol=1e-6):
    """Returns 0 if A is Hermitian symmetric to within tol

    Parameters
    ----------
    A   : {dense or sparse matrix}
        e.g. array, matrix, csr_matrix, ...
    fast_check : {bool}
        If True, use the heuristic < Ax, y> = < x, Ay>
        for random vectors x and y to check for symmetry.
        If False, compute A - A.H.
    tol : {float}
        Symmetry tolerance

    Returns
    -------
    0                        if symmetric
    max( \|A - A.H\| )       if unsymmetric and fast_check=False
    abs( <Ax, y> - <x, Ay> ) if unsymmetric and fast_check=False

    Notes
    -----
    This function applies a simple test of Hermitian symmetry

    Examples
    --------
    >>> import numpy
    >>> from pyamg.util.linalg import issymm
    >>> issymm(numpy.array([[1,2],[1,1]]))
    1

    >>> from pyamg.gallery import poisson
    >>> issymm(poisson((10,10)))
    0

    """
    # convert to matrix type
    if not sparse.isspmatrix(A):
        A = numpy.asmatrix(A)

    if fast_check:
        x = scipy.rand(A.shape[0], 1)
        y = scipy.rand(A.shape[0], 1)
        if A.dtype == complex:
            x += 1.0j * scipy.rand(A.shape[0], 1)
            y += 1.0j * scipy.rand(A.shape[0], 1)
        diff = float(
            numpy.abs(
                numpy.dot((A * x).conjugate().T, y) - numpy.dot(x.conjugate().T, A * y)
            )
        )

    else:
        # compute the difference, A - A.H
        if sparse.isspmatrix(A):
            diff = numpy.ravel((A - A.H).data)
        else:
            diff = numpy.ravel(A - A.H)

        if numpy.max(diff.shape) == 0:
            diff = 0
        else:
            diff = numpy.max(numpy.abs(diff))

    if diff < tol:
        diff = 0

    return diff


def pinv_array(a, cond=None):
    """Calculate the Moore-Penrose pseudo inverse of each block of
        the three dimensional array a.

    Parameters
    ----------
    a   : {dense array}
        Is of size (n, m, m)
    cond : {float}
        Used by *gelss to filter numerically zeros singular values.
        If None, a suitable value is chosen for you.

    Returns
    -------
    Nothing, a is modified in place so that a[k] holds the pseudoinverse
    of that block.

    Notes
    -----
    By using lapack wrappers, this can be much faster for large n, than
    directly calling pinv2

    Examples
    --------
    >>> import numpy
    >>> from pyamg.util.linalg import pinv_array
    >>> a = numpy.array([[[1.,2.],[1.,1.]], [[1.,1.],[3.,3.]]])
    >>> ac = a.copy()
    >>> pinv_array(a)
    >>> print numpy.dot(a[0], ac[0])
    [[  1.00000000e+00  -2.22044605e-16]
     [  4.44089210e-16   1.00000000e+00]]

    >>>print(numpy.dot(a[1], ac[1]))
    [[ 0.5  0.5]
     [ 0.5  0.5]]

    """

    raise RuntimeError(
        "lapack not available. Please comment line +++from scipy.linalg.lapack import get_lapack_funcs+++"
    )
    n = a.shape[0]
    m = a.shape[1]

    if m == 1:
        ##
        # Pseudo-inverse of 1 x 1 matrices is trivial
        zero_entries = (a == 0.0).nonzero()[0]
        a[:] = 1.0 / a
        a[zero_entries] = 0.0
        del zero_entries

    else:
        ##
        # The block size is greater than 1

        ##
        # Create necessary arrays and function pointers for calculating pinv
        (gelss,) = get_lapack_funcs(("gelss",), (numpy.ones((1,), dtype=a.dtype)))
        RHS = numpy.eye(m, dtype=a.dtype)
        lwork = calc_lwork.gelss(gelss.prefix, m, m, m)[1]

        ##
        # Choose tolerance for which singular values are zero in *gelss below
        if cond is None:
            t = a.dtype.char
            eps = numpy.finfo(numpy.float).eps
            feps = numpy.finfo(numpy.single).eps
            geps = numpy.finfo(numpy.longfloat).eps
            _array_precision = {"f": 0, "d": 1, "g": 2, "F": 0, "D": 1, "G": 2}
            cond = {0: feps * 1e3, 1: eps * 1e6, 2: geps * 1e6}[_array_precision[t]]

        ##
        # Invert each block of a
        for kk in xrange(n):
            v, a[kk], s, rank, info = gelss(
                a[kk], RHS, cond=cond, lwork=lwork, overwrite_a=True, overwrite_b=False
            )
