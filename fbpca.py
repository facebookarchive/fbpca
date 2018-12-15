"""
Functions for principal component analysis (PCA) and accuracy checks

---------------------------------------------------------------------

This module contains eight functions:

pca
    principal component analysis (singular value decomposition)
eigens
    eigendecomposition of a self-adjoint matrix
eigenn
    eigendecomposition of a nonnegative-definite self-adjoint matrix
diffsnorm
    spectral-norm accuracy of a singular value decomposition
diffsnormc
    spectral-norm accuracy of a centered singular value decomposition
diffsnorms
    spectral-norm accuracy of a Schur decomposition
mult
    default matrix multiplication
set_matrix_mult
    re-definition of the matrix multiplication function "mult"

---------------------------------------------------------------------

Copyright 2018 Facebook Inc.
All rights reserved.

"Software" means the fbpca software distributed by Facebook Inc.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:

* Redistributions of source code must retain the above copyright
  notice, this list of conditions, and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright
  notice, this list of conditions, and the following disclaimer in
  the documentation and/or other materials provided with the
  distribution.

* Neither the name Facebook nor the names of its contributors may be
  used to endorse or promote products derived from this software
  without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED
AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY
WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.

Additional grant of patent rights:

Facebook hereby grants you a perpetual, worldwide, royalty-free,
non-exclusive, irrevocable (subject to the termination provision
below) license under any rights in any patent claims owned by
Facebook, to make, have made, use, sell, offer to sell, import, and
otherwise transfer the Software. For avoidance of doubt, no license
is granted under Facebook's rights in any patent claims that are
infringed by (i) modifications to the Software made by you or a third
party, or (ii) the Software in combination with any software or other
technology provided by you or a third party.

The license granted hereunder will terminate, automatically and
without notice, for anyone that makes any claim (including by filing
any lawsuit, assertion, or other action) alleging (a) direct,
indirect, or contributory infringement or inducement to infringe any
patent: (i) by Facebook or any of its subsidiaries or affiliates,
whether or not such claim is related to the Software, (ii) by any
party if such claim arises in whole or in part from any software,
product or service of Facebook or any of its subsidiaries or
affiliates, whether or not such claim is related to the Software, or
(iii) by any party relating to the Software; or (b) that any right in
any patent claim of Facebook is invalid or unenforceable.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import math

import numpy as np
from scipy.linalg import cholesky, eigh, lu, qr, svd, norm, solve
from scipy.sparse import issparse


def diffsnorm(A, U, s, Va, n_iter=20):
    """
    2-norm accuracy of an approx to a matrix.

    Computes an estimate snorm of the spectral norm (the operator norm
    induced by the Euclidean vector norm) of A - U diag(s) Va, using
    n_iter iterations of the power method started with a random vector;
    n_iter must be a positive integer.

    Increasing n_iter improves the accuracy of the estimate snorm of
    the spectral norm of A - U diag(s) Va.

    Notes
    -----
    To obtain repeatable results, reset the seed for the pseudorandom
    number generator.

    Parameters
    ----------
    A : array_like
        first matrix in A - U diag(s) Va whose spectral norm is being
        estimated
    U : array_like
        second matrix in A - U diag(s) Va whose spectral norm is being
        estimated
    s : array_like
        vector in A - U diag(s) Va whose spectral norm is being
        estimated
    Va : array_like
        fourth matrix in A - U diag(s) Va whose spectral norm is being
        estimated
    n_iter : int, optional
        number of iterations of the power method to conduct;
        n_iter must be a positive integer, and defaults to 20

    Returns
    -------
    float
        an estimate of the spectral norm of A - U diag(s) Va (the
        estimate fails to be accurate with exponentially low prob. as
        n_iter increases; see references DM1_, DM2_, and DM3_ below)

    Examples
    --------
    >>> from fbpca import diffsnorm, pca
    >>> from numpy.random import uniform
    >>> from scipy.linalg import svd
    >>>
    >>> A = uniform(low=-1.0, high=1.0, size=(100, 2))
    >>> A = A.dot(uniform(low=-1.0, high=1.0, size=(2, 100)))
    >>> (U, s, Va) = svd(A, full_matrices=False)
    >>> A = A / s[0]
    >>>
    >>> (U, s, Va) = pca(A, 2, True)
    >>> err = diffsnorm(A, U, s, Va)
    >>> print(err)

    This example produces a rank-2 approximation U diag(s) Va to A such
    that the columns of U are orthonormal, as are the rows of Va, and
    the entries of s are all nonnegative and are nonincreasing.
    diffsnorm(A, U, s, Va) outputs an estimate of the spectral norm of
    A - U diag(s) Va, which should be close to the machine precision.

    References
    ----------
    .. [DM1] Jacek Kuczynski and Henryk Wozniakowski, Estimating the
             largest eigenvalues by the power and Lanczos methods with
             a random start, SIAM Journal on Matrix Analysis and
             Applications, 13 (4): 1094-1122, 1992.
    .. [DM2] Edo Liberty, Franco Woolfe, Per-Gunnar Martinsson,
             Vladimir Rokhlin, and Mark Tygert, Randomized algorithms
             for the low-rank approximation of matrices, Proceedings of
             the National Academy of Sciences (USA), 104 (51):
             20167-20172, 2007. (See the appendix.)
    .. [DM3] Franco Woolfe, Edo Liberty, Vladimir Rokhlin, and Mark
             Tygert, A fast randomized algorithm for the approximation
             of matrices, Applied and Computational Harmonic Analysis,
             25 (3): 335-366, 2008. (See Section 3.4.)

    See also
    --------
    diffsnormc, pca
    """

    (m, n) = A.shape
    (m2, k) = U.shape
    k2 = len(s)
    l = len(s)
    (l2, n2) = Va.shape

    assert m == m2
    assert k == k2
    assert l == l2
    assert n == n2

    assert n_iter >= 1

    if np.isrealobj(A) and np.isrealobj(U) and np.isrealobj(s) and np.isrealobj(Va):
        isreal = True
    else:
        isreal = False

    # Promote the types of integer data to float data.
    dtype = (A * 1.0).dtype

    if m >= n:

        #
        # Generate a random vector x.
        #
        if isreal:
            x = np.random.normal(size=(n, 1)).astype(dtype)
        else:
            x = np.random.normal(size=(n, 1)).astype(dtype) + 1j * np.random.normal(
                size=(n, 1)
            ).astype(dtype)

        x = x / norm(x)

        #
        # Run n_iter iterations of the power method.
        #
        for it in range(n_iter):
            #
            # Set y = (A - U diag(s) Va)x.
            #
            y = mult(A, x) - U.dot(np.diag(s).dot(Va.dot(x)))
            #
            # Set x = (A' - Va' diag(s)' U')y.
            #
            x = mult(y.conj().T, A).conj().T - Va.conj().T.dot(
                np.diag(s).conj().T.dot(U.conj().T.dot(y))
            )

            #
            # Normalize x, memorizing its Euclidean norm.
            #
            snorm = norm(x)
            if snorm == 0:
                return 0
            x = x / snorm

        snorm = math.sqrt(snorm)

    if m < n:

        #
        # Generate a random vector y.
        #
        if isreal:
            y = np.random.normal(size=(m, 1)).astype(dtype)
        else:
            y = np.random.normal(size=(m, 1)).astype(dtype) + 1j * np.random.normal(
                size=(m, 1)
            ).astype(dtype)

        y = y / norm(y)

        #
        # Run n_iter iterations of the power method.
        #
        for it in range(n_iter):
            #
            # Set x = (A' - Va' diag(s)' U')y.
            #
            x = mult(y.conj().T, A).conj().T - Va.conj().T.dot(
                np.diag(s).conj().T.dot(U.conj().T.dot(y))
            )
            #
            # Set y = (A - U diag(s) Va)x.
            #
            y = mult(A, x) - U.dot(np.diag(s).dot(Va.dot(x)))

            #
            # Normalize y, memorizing its Euclidean norm.
            #
            snorm = norm(y)
            if snorm == 0:
                return 0
            y = y / snorm

        snorm = math.sqrt(snorm)

    return snorm


def diffsnormc(A, U, s, Va, n_iter=20):
    """
    2-norm approx error to a matrix upon centering.

    Computes an estimate snorm of the spectral norm (the operator norm
    induced by the Euclidean vector norm) of C(A) - U diag(s) Va, using
    n_iter iterations of the power method started with a random vector,
    where C(A) refers to A from the input, after centering its columns;
    n_iter must be a positive integer.

    Increasing n_iter improves the accuracy of the estimate snorm of
    the spectral norm of C(A) - U diag(s) Va, where C(A) refers to A
    after centering its columns.

    Notes
    -----
    To obtain repeatable results, reset the seed for the pseudorandom
    number generator.

    Parameters
    ----------
    A : array_like
        first matrix in the column-centered A - U diag(s) Va whose
        spectral norm is being estimated
    U : array_like
        second matrix in the column-centered A - U diag(s) Va whose
        spectral norm is being estimated
    s : array_like
        vector in the column-centered A - U diag(s) Va whose spectral
        norm is being estimated
    Va : array_like
        fourth matrix in the column-centered A - U diag(s) Va whose
        spectral norm is being estimated
    n_iter : int, optional
        number of iterations of the power method to conduct;
        n_iter must be a positive integer, and defaults to 20

    Returns
    -------
    float
        an estimate of the spectral norm of the column-centered A
        - U diag(s) Va (the estimate fails to be accurate with
        exponentially low probability as n_iter increases; see
        references DC1_, DC2_, and DC3_ below)

    Examples
    --------
    >>> from fbpca import diffsnormc, pca
    >>> from numpy.random import uniform
    >>> from scipy.linalg import svd
    >>>
    >>> A = uniform(low=-1.0, high=1.0, size=(100, 2))
    >>> A = A.dot(uniform(low=-1.0, high=1.0, size=(2, 100)))
    >>> (U, s, Va) = svd(A, full_matrices=False)
    >>> A = A / s[0]
    >>>
    >>> (U, s, Va) = pca(A, 2, False)
    >>> err = diffsnormc(A, U, s, Va)
    >>> print(err)

    This example produces a rank-2 approximation U diag(s) Va to the
    column-centered A such that the columns of U are orthonormal, as
    are the rows of Va, and the entries of s are nonnegative and
    nonincreasing. diffsnormc(A, U, s, Va) outputs an estimate of the
    spectral norm of the column-centered A - U diag(s) Va, which
    should be close to the machine precision.

    References
    ----------
    .. [DC1] Jacek Kuczynski and Henryk Wozniakowski, Estimating the
             largest eigenvalues by the power and Lanczos methods with
             a random start, SIAM Journal on Matrix Analysis and
             Applications, 13 (4): 1094-1122, 1992.
    .. [DC2] Edo Liberty, Franco Woolfe, Per-Gunnar Martinsson,
             Vladimir Rokhlin, and Mark Tygert, Randomized algorithms
             for the low-rank approximation of matrices, Proceedings of
             the National Academy of Sciences (USA), 104 (51):
             20167-20172, 2007. (See the appendix.)
    .. [DC3] Franco Woolfe, Edo Liberty, Vladimir Rokhlin, and Mark
             Tygert, A fast randomized algorithm for the approximation
             of matrices, Applied and Computational Harmonic Analysis,
             25 (3): 335-366, 2008. (See Section 3.4.)

    See also
    --------
    diffsnorm, pca
    """

    (m, n) = A.shape
    (m2, k) = U.shape
    k2 = len(s)
    l = len(s)
    (l2, n2) = Va.shape

    assert m == m2
    assert k == k2
    assert l == l2
    assert n == n2

    assert n_iter >= 1

    if np.isrealobj(A) and np.isrealobj(U) and np.isrealobj(s) and np.isrealobj(Va):
        isreal = True
    else:
        isreal = False

    # Promote the types of integer data to float data.
    dtype = (A * 1.0).dtype

    #
    # Calculate the average of the entries in every column.
    #
    c = A.sum(axis=0) / m
    c = c.reshape((1, n))

    if m >= n:

        #
        # Generate a random vector x.
        #
        if isreal:
            x = np.random.normal(size=(n, 1)).astype(dtype)
        else:
            x = np.random.normal(size=(n, 1)).astype(dtype) + 1j * np.random.normal(
                size=(n, 1)
            ).astype(dtype)

        x = x / norm(x)

        #
        # Run n_iter iterations of the power method.
        #
        for it in range(n_iter):
            #
            # Set y = (A - ones(m,1)*c - U diag(s) Va)x.
            #
            y = (
                mult(A, x)
                - np.ones((m, 1), dtype=dtype).dot(c.dot(x))
                - U.dot(np.diag(s).dot(Va.dot(x)))
            )
            #
            # Set x = (A' - c'*ones(1,m) - Va' diag(s)' U')y.
            #
            x = (
                mult(y.conj().T, A).conj().T
                - c.conj().T.dot(np.ones((1, m), dtype=dtype).dot(y))
                - Va.conj().T.dot(np.diag(s).conj().T.dot(U.conj().T.dot(y)))
            )

            #
            # Normalize x, memorizing its Euclidean norm.
            #
            snorm = norm(x)
            if snorm == 0:
                return 0
            x = x / snorm

        snorm = math.sqrt(snorm)

    if m < n:

        #
        # Generate a random vector y.
        #
        if isreal:
            y = np.random.normal(size=(m, 1)).astype(dtype)
        else:
            y = np.random.normal(size=(m, 1)).astype(dtype) + 1j * np.random.normal(
                size=(m, 1)
            ).astype(dtype)

        y = y / norm(y)

        #
        # Run n_iter iterations of the power method.
        #
        for it in range(n_iter):
            #
            # Set x = (A' - c'*ones(1,m) - Va' diag(s)' U')y.
            #
            x = (
                mult(y.conj().T, A).conj().T
                - c.conj().T.dot(np.ones((1, m), dtype=dtype).dot(y))
                - Va.conj().T.dot(np.diag(s).conj().T.dot(U.conj().T.dot(y)))
            )
            #
            # Set y = (A - ones(m,1)*c - U diag(s) Va)x.
            #
            y = (
                mult(A, x)
                - np.ones((m, 1), dtype=dtype).dot(c.dot(x))
                - U.dot(np.diag(s).dot(Va.dot(x)))
            )

            #
            # Normalize y, memorizing its Euclidean norm.
            #
            snorm = norm(y)
            if snorm == 0:
                return 0
            y = y / snorm

        snorm = math.sqrt(snorm)

    return snorm


def diffsnorms(A, S, V, n_iter=20):
    """
    2-norm accuracy of a Schur decomp. of a matrix.

    Computes an estimate snorm of the spectral norm (the operator norm
    induced by the Euclidean vector norm) of A-VSV', using n_iter
    iterations of the power method started with a random vector;
    n_iter must be a positive integer.

    Increasing n_iter improves the accuracy of the estimate snorm of
    the spectral norm of A-VSV'.

    Notes
    -----
    To obtain repeatable results, reset the seed for the pseudorandom
    number generator.

    Parameters
    ----------
    A : array_like
        first matrix in A-VSV' whose spectral norm is being estimated
    S : array_like
        third matrix in A-VSV' whose spectral norm is being estimated
    V : array_like
        second matrix in A-VSV' whose spectral norm is being estimated
    n_iter : int, optional
        number of iterations of the power method to conduct;
        n_iter must be a positive integer, and defaults to 20

    Returns
    -------
    float
        an estimate of the spectral norm of A-VSV' (the estimate fails
        to be accurate with exponentially low probability as n_iter
        increases; see references DS1_, DS2_, and DS3_ below)

    Examples
    --------
    >>> from fbpca import diffsnorms, eigenn
    >>> from numpy import diag
    >>> from numpy.random import uniform
    >>> from scipy.linalg import svd
    >>>
    >>> A = uniform(low=-1.0, high=1.0, size=(2, 100))
    >>> A = A.T.dot(A)
    >>> (U, s, Va) = svd(A, full_matrices=False)
    >>> A = A / s[0]
    >>>
    >>> (d, V) = eigenn(A, 2)
    >>> err = diffsnorms(A, diag(d), V)
    >>> print(err)

    This example produces a rank-2 approximation V diag(d) V' to A
    such that the columns of V are orthonormal and the entries of d
    are nonnegative and are nonincreasing.
    diffsnorms(A, diag(d), V) outputs an estimate of the spectral norm
    of A - V diag(d) V', which should be close to the machine
    precision.

    References
    ----------
    .. [DS1] Jacek Kuczynski and Henryk Wozniakowski, Estimating the
             largest eigenvalues by the power and Lanczos methods with
             a random start, SIAM Journal on Matrix Analysis and
             Applications, 13 (4): 1094-1122, 1992.
    .. [DS2] Edo Liberty, Franco Woolfe, Per-Gunnar Martinsson,
             Vladimir Rokhlin, and Mark Tygert, Randomized algorithms
             for the low-rank approximation of matrices, Proceedings of
             the National Academy of Sciences (USA), 104 (51):
             20167-20172, 2007. (See the appendix.)
    .. [DS3] Franco Woolfe, Edo Liberty, Vladimir Rokhlin, and Mark
             Tygert, A fast randomized algorithm for the approximation
             of matrices, Applied and Computational Harmonic Analysis,
             25 (3): 335-366, 2008. (See Section 3.4.)

    See also
    --------
    eigenn, eigens
    """

    (m, n) = A.shape
    (m2, k) = V.shape
    (k2, k3) = S.shape

    assert m == n
    assert m == m2
    assert k == k2
    assert k2 == k3

    assert n_iter >= 1

    if np.isrealobj(A) and np.isrealobj(V) and np.isrealobj(S):
        isreal = True
    else:
        isreal = False

    # Promote the types of integer data to float data.
    dtype = (A * 1.0).dtype

    #
    # Generate a random vector x.
    #
    if isreal:
        x = np.random.normal(size=(n, 1)).astype(dtype)
    else:
        x = np.random.normal(size=(n, 1)).astype(dtype) + 1j * np.random.normal(
            size=(n, 1)
        ).astype(dtype)

    x = x / norm(x)

    #
    # Run n_iter iterations of the power method.
    #
    for it in range(n_iter):
        #
        # Set y = (A-VSV')x.
        #
        y = mult(A, x) - V.dot(S.dot(V.conj().T.dot(x)))
        #
        # Set x = (A'-VS'V')y.
        #
        x = mult(y.conj().T, A).conj().T - V.dot(S.conj().T.dot(V.conj().T.dot(y)))

        #
        # Normalize x, memorizing its Euclidean norm.
        #
        snorm = norm(x)
        if snorm == 0:
            return 0
        x = x / snorm

    snorm = math.sqrt(snorm)

    return snorm


def eigenn(A, k=6, n_iter=4, l=None):
    """
    Eigendecomposition of a NONNEGATIVE-DEFINITE matrix.

    Constructs a nearly optimal rank-k approximation V diag(d) V' to a
    NONNEGATIVE-DEFINITE matrix A, using n_iter normalized power
    iterations, with block size l, started with an n x l random matrix,
    when A is n x n; the reference EGN_ below explains "nearly
    optimal." k must be a positive integer <= the dimension n of A,
    n_iter must be a nonnegative integer, and l must be a positive
    integer >= k.

    The rank-k approximation V diag(d) V' comes in the form of an
    eigendecomposition -- the columns of V are orthonormal and d is a
    real vector such that its entries are nonnegative and nonincreasing.
    V is n x k and len(d) = k, when A is n x n.

    Increasing n_iter or l improves the accuracy of the approximation
    V diag(d) V'; the reference EGN_ below describes how the accuracy
    depends on n_iter and l. Please note that even n_iter=1 guarantees
    superb accuracy, whether or not there is any gap in the singular
    values of the matrix A being approximated, at least when measuring
    accuracy as the spectral norm || A - V diag(d) V' || of the matrix
    A - V diag(d) V' (relative to the spectral norm ||A|| of A).

    Notes
    -----
    THE MATRIX A MUST BE SELF-ADJOINT AND NONNEGATIVE DEFINITE.

    To obtain repeatable results, reset the seed for the pseudorandom
    number generator.

    The user may ascertain the accuracy of the approximation
    V diag(d) V' to A by invoking diffsnorms(A, numpy.diag(d), V).

    Parameters
    ----------
    A : array_like, shape (n, n)
        matrix being approximated
    k : int, optional
        rank of the approximation being constructed;
        k must be a positive integer <= the dimension of A, and
        defaults to 6
    n_iter : int, optional
        number of normalized power iterations to conduct;
        n_iter must be a nonnegative integer, and defaults to 4
    l : int, optional
        block size of the normalized power iterations;
        l must be a positive integer >= k, and defaults to k+2

    Returns
    -------
    d : ndarray, shape (k,)
        vector of length k in the rank-k approximation V diag(d) V'
        to A, such that its entries are nonnegative and nonincreasing
    V : ndarray, shape (n, k)
        n x k matrix in the rank-k approximation V diag(d) V' to A,
        where A is n x n

    Examples
    --------
    >>> from fbpca import diffsnorms, eigenn
    >>> from numpy import diag
    >>> from numpy.random import uniform
    >>> from scipy.linalg import svd
    >>>
    >>> A = uniform(low=-1.0, high=1.0, size=(2, 100))
    >>> A = A.T.dot(A)
    >>> (U, s, Va) = svd(A, full_matrices=False)
    >>> A = A / s[0]
    >>>
    >>> (d, V) = eigenn(A, 2)
    >>> err = diffsnorms(A, diag(d), V)
    >>> print(err)

    This example produces a rank-2 approximation V diag(d) V' to A
    such that the columns of V are orthonormal and the entries of d
    are nonnegative and nonincreasing.
    diffsnorms(A, diag(d), V) outputs an estimate of the spectral norm
    of A - V diag(d) V', which should be close to the machine
    precision.

    References
    ----------
    .. [EGN] Nathan Halko, Per-Gunnar Martinsson, and Joel Tropp,
             Finding structure with randomness: probabilistic
             algorithms for constructing approximate matrix
             decompositions, arXiv:0909.4061 [math.NA; math.PR], 2009
             (available at `arXiv <http://arxiv.org/abs/0909.4061>`_).

    See also
    --------
    diffsnorms, eigens, pca
    """

    if l is None:
        l = k + 2

    (m, n) = A.shape

    assert m == n
    assert k > 0
    assert k <= n
    assert n_iter >= 0
    assert l >= k

    if np.isrealobj(A):
        isreal = True
    else:
        isreal = False

    # Promote the types of integer data to float data.
    dtype = (A * 1.0).dtype

    #
    # Check whether A is self-adjoint to nearly the machine precision.
    #
    x = np.random.uniform(low=-1.0, high=1.0, size=(n, 1)).astype(dtype)
    y = mult(A, x)
    z = mult(x.conj().T, A).conj().T
    if dtype == "float16":
        prec = 0.1e-1
    elif dtype in ["float32", "complex64"]:
        prec = 0.1e-3
    else:
        prec = 0.1e-11
    assert (norm(y - z) <= prec * norm(y)) and (norm(y - z) <= prec * norm(z))

    #
    # Eigendecompose A directly if l >= n/1.25.
    #
    if l >= n / 1.25:
        (d, V) = eigh(A.todense() if issparse(A) else A)
        #
        # Retain only the entries of d with the k greatest absolute
        # values and the corresponding columns of V.
        #
        idx = abs(d).argsort()[-k:][::-1]
        return abs(d[idx]), V[:, idx]

    #
    # Apply A to a random matrix, obtaining Q.
    #
    if isreal:
        R = np.random.uniform(low=-1.0, high=1.0, size=(n, l)).astype(dtype)
    if not isreal:
        R = np.random.uniform(low=-1.0, high=1.0, size=(n, l)).astype(dtype)
        R += 1j * np.random.uniform(low=-1.0, high=1.0, size=(n, l)).astype(dtype)

    Q = mult(A, R)

    #
    # Form a matrix Q whose columns constitute a well-conditioned basis
    # for the columns of the earlier Q.
    #
    if n_iter == 0:

        anorm = 0
        for j in range(l):
            anorm = max(anorm, norm(Q[:, j]) / norm(R[:, j]))

        (Q, _) = qr(Q, mode="economic")

    if n_iter > 0:

        (Q, _) = lu(Q, permute_l=True)

    #
    # Conduct normalized power iterations.
    #
    for it in range(n_iter):

        cnorm = np.zeros((l), dtype=dtype)
        for j in range(l):
            cnorm[j] = norm(Q[:, j])

        Q = mult(A, Q)

        if it + 1 < n_iter:

            (Q, _) = lu(Q, permute_l=True)

        else:

            anorm = 0
            for j in range(l):
                anorm = max(anorm, norm(Q[:, j]) / cnorm[j])

            (Q, _) = qr(Q, mode="economic")

    #
    # Use the Nystrom method to obtain approximations to the
    # eigenvalues and eigenvectors of A (shifting A on the subspace
    # spanned by the columns of Q in order to make the shifted A be
    # positive definite). An alternative is to use the (symmetric)
    # square root in place of the Cholesky factor of the shift.
    #
    anorm = 0.1e-6 * anorm * math.sqrt(1.0 * n)
    E = mult(A, Q) + anorm * Q
    R = Q.conj().T.dot(E)
    R = (R + R.conj().T) / 2
    R = cholesky(R, lower=True)
    (E, d, V) = svd(solve(R, E.conj().T), full_matrices=False)
    V = V.conj().T
    d = d * d - anorm

    #
    # Retain only the entries of d with the k greatest absolute values
    # and the corresponding columns of V.
    #
    idx = abs(d).argsort()[-k:][::-1]
    return abs(d[idx]), V[:, idx]


def eigens(A, k=6, n_iter=4, l=None):
    """
    Eigendecomposition of a SELF-ADJOINT matrix.

    Constructs a nearly optimal rank-k approximation V diag(d) V' to a
    SELF-ADJOINT matrix A, using n_iter normalized power iterations,
    with block size l, started with an n x l random matrix, when A is
    n x n; the reference EGS_ below explains "nearly optimal." k must
    be a positive integer <= the dimension n of A, n_iter must be a
    nonnegative integer, and l must be a positive integer >= k.

    The rank-k approximation V diag(d) V' comes in the form of an
    eigendecomposition -- the columns of V are orthonormal and d is a
    vector whose entries are real-valued and their absolute values are
    nonincreasing. V is n x k and len(d) = k, when A is n x n.

    Increasing n_iter or l improves the accuracy of the approximation
    V diag(d) V'; the reference EGS_ below describes how the accuracy
    depends on n_iter and l. Please note that even n_iter=1 guarantees
    superb accuracy, whether or not there is any gap in the singular
    values of the matrix A being approximated, at least when measuring
    accuracy as the spectral norm || A - V diag(d) V' || of the matrix
    A - V diag(d) V' (relative to the spectral norm ||A|| of A).

    Notes
    -----
    THE MATRIX A MUST BE SELF-ADJOINT.

    To obtain repeatable results, reset the seed for the pseudorandom
    number generator.

    The user may ascertain the accuracy of the approximation
    V diag(d) V' to A by invoking diffsnorms(A, numpy.diag(d), V).

    Parameters
    ----------
    A : array_like, shape (n, n)
        matrix being approximated
    k : int, optional
        rank of the approximation being constructed;
        k must be a positive integer <= the dimension of A, and
        defaults to 6
    n_iter : int, optional
        number of normalized power iterations to conduct;
        n_iter must be a nonnegative integer, and defaults to 4
    l : int, optional
        block size of the normalized power iterations;
        l must be a positive integer >= k, and defaults to k+2

    Returns
    -------
    d : ndarray, shape (k,)
        vector of length k in the rank-k approximation V diag(d) V'
        to A, such that its entries are real-valued and their absolute
        values are nonincreasing
    V : ndarray, shape (n, k)
        n x k matrix in the rank-k approximation V diag(d) V' to A,
        where A is n x n

    Examples
    --------
    >>> from fbpca import diffsnorms, eigens
    >>> from numpy import diag
    >>> from numpy.random import uniform
    >>> from scipy.linalg import svd
    >>>
    >>> A = uniform(low=-1.0, high=1.0, size=(2, 100))
    >>> A = A.T.dot(A)
    >>> (U, s, Va) = svd(A, full_matrices=False)
    >>> A = A / s[0]
    >>>
    >>> (d, V) = eigens(A, 2)
    >>> err = diffsnorms(A, diag(d), V)
    >>> print(err)

    This example produces a rank-2 approximation V diag(d) V' to A
    such that the columns of V are orthonormal, and the entries of d
    are real-valued and their absolute values are nonincreasing.
    diffsnorms(A, diag(d), V) outputs an estimate of the spectral norm
    of A - V diag(d) V', which should be close to the machine
    precision.

    References
    ----------
    .. [EGS] Nathan Halko, Per-Gunnar Martinsson, and Joel Tropp,
             Finding structure with randomness: probabilistic
             algorithms for constructing approximate matrix
             decompositions, arXiv:0909.4061 [math.NA; math.PR], 2009
             (available at `arXiv <http://arxiv.org/abs/0909.4061>`_).

    See also
    --------
    diffsnorms, eigenn, pca
    """

    if l is None:
        l = k + 2

    (m, n) = A.shape

    assert m == n
    assert k > 0
    assert k <= n
    assert n_iter >= 0
    assert l >= k

    if np.isrealobj(A):
        isreal = True
    else:
        isreal = False

    # Promote the types of integer data to float data.
    dtype = (A * 1.0).dtype

    #
    # Check whether A is self-adjoint to nearly the machine precision.
    #
    x = np.random.uniform(low=-1.0, high=1.0, size=(n, 1)).astype(dtype)
    y = mult(A, x)
    z = mult(x.conj().T, A).conj().T
    if dtype == "float16":
        prec = 0.1e-1
    elif dtype in ["float32", "complex64"]:
        prec = 0.1e-3
    else:
        prec = 0.1e-11
    assert (norm(y - z) <= prec * norm(y)) and (norm(y - z) <= prec * norm(z))

    #
    # Eigendecompose A directly if l >= n/1.25.
    #
    if l >= n / 1.25:
        (d, V) = eigh(A.todense() if issparse(A) else A)
        #
        # Retain only the entries of d with the k greatest absolute
        # values and the corresponding columns of V.
        #
        idx = abs(d).argsort()[-k:][::-1]
        return d[idx], V[:, idx]

    #
    # Apply A to a random matrix, obtaining Q.
    #
    if isreal:
        Q = np.random.uniform(low=-1.0, high=1.0, size=(n, l)).astype(dtype)
        Q = mult(A, Q)
    if not isreal:
        Q = np.random.uniform(low=-1.0, high=1.0, size=(n, l)).astype(dtype)
        Q = Q + 1j * np.random.uniform(low=-1.0, high=1.0, size=(n, l)).astype(dtype)
        Q = mult(A, Q)

    #
    # Form a matrix Q whose columns constitute a well-conditioned basis
    # for the columns of the earlier Q.
    #
    if n_iter == 0:
        (Q, _) = qr(Q, mode="economic")
    if n_iter > 0:
        (Q, _) = lu(Q, permute_l=True)

    #
    # Conduct normalized power iterations.
    #
    for it in range(n_iter):

        Q = mult(A, Q)

        if it + 1 < n_iter:
            (Q, _) = lu(Q, permute_l=True)
        else:
            (Q, _) = qr(Q, mode="economic")

    #
    # Eigendecompose Q'*A*Q to obtain approximations to the eigenvalues
    # and eigenvectors of A.
    #
    R = Q.conj().T.dot(mult(A, Q))
    R = (R + R.conj().T) / 2
    (d, V) = eigh(R)
    V = Q.dot(V)

    #
    # Retain only the entries of d with the k greatest absolute values
    # and the corresponding columns of V.
    #
    idx = abs(d).argsort()[-k:][::-1]
    return d[idx], V[:, idx]


def pca(A, k=6, raw=False, n_iter=2, l=None):
    """
    Principal component analysis.

    Constructs a nearly optimal rank-k approximation U diag(s) Va to A,
    centering the columns of A first when raw is False, using n_iter
    normalized power iterations, with block size l, started with a
    min(m,n) x l random matrix, when A is m x n; the reference PCA_
    below explains "nearly optimal." k must be a positive integer <=
    the smaller dimension of A, n_iter must be a nonnegative integer,
    and l must be a positive integer >= k.

    The rank-k approximation U diag(s) Va comes in the form of a
    singular value decomposition (SVD) -- the columns of U are
    orthonormal, as are the rows of Va, and the entries of s are all
    nonnegative and nonincreasing. U is m x k, Va is k x n, and
    len(s)=k, when A is m x n.

    Increasing n_iter or l improves the accuracy of the approximation
    U diag(s) Va; the reference PCA_ below describes how the accuracy
    depends on n_iter and l. Please note that even n_iter=1 guarantees
    superb accuracy, whether or not there is any gap in the singular
    values of the matrix A being approximated, at least when measuring
    accuracy as the spectral norm || A - U diag(s) Va || of the matrix
    A - U diag(s) Va (relative to the spectral norm ||A|| of A, and
    accounting for centering when raw is False).

    Notes
    -----
    To obtain repeatable results, reset the seed for the pseudorandom
    number generator.

    The user may ascertain the accuracy of the approximation
    U diag(s) Va to A by invoking diffsnorm(A, U, s, Va), when raw is
    True. The user may ascertain the accuracy of the approximation
    U diag(s) Va to C(A), where C(A) refers to A after centering its
    columns, by invoking diffsnormc(A, U, s, Va), when raw is False.

    Parameters
    ----------
    A : array_like, shape (m, n)
        matrix being approximated
    k : int, optional
        rank of the approximation being constructed;
        k must be a positive integer <= the smaller dimension of A,
        and defaults to 6
    raw : bool, optional
        centers A when raw is False but does not when raw is True;
        raw must be a Boolean and defaults to False
    n_iter : int, optional
        number of normalized power iterations to conduct;
        n_iter must be a nonnegative integer, and defaults to 2
    l : int, optional
        block size of the normalized power iterations;
        l must be a positive integer >= k, and defaults to k+2

    Returns
    -------
    U : ndarray, shape (m, k)
        m x k matrix in the rank-k approximation U diag(s) Va to A or
        C(A), where A is m x n, and C(A) refers to A after centering
        its columns; the columns of U are orthonormal
    s : ndarray, shape (k,)
        vector of length k in the rank-k approximation U diag(s) Va to
        A or C(A), where A is m x n, and C(A) refers to A after
        centering its columns; the entries of s are all nonnegative and
        nonincreasing
    Va : ndarray, shape (k, n)
        k x n matrix in the rank-k approximation U diag(s) Va to A or
        C(A), where A is m x n, and C(A) refers to A after centering
        its columns; the rows of Va are orthonormal

    Examples
    --------
    >>> from fbpca import diffsnorm, pca
    >>> from numpy.random import uniform
    >>> from scipy.linalg import svd
    >>>
    >>> A = uniform(low=-1.0, high=1.0, size=(100, 2))
    >>> A = A.dot(uniform(low=-1.0, high=1.0, size=(2, 100)))
    >>> (U, s, Va) = svd(A, full_matrices=False)
    >>> A = A / s[0]
    >>>
    >>> (U, s, Va) = pca(A, 2, True)
    >>> err = diffsnorm(A, U, s, Va)
    >>> print(err)

    This example produces a rank-2 approximation U diag(s) Va to A such
    that the columns of U are orthonormal, as are the rows of Va, and
    the entries of s are all nonnegative and are nonincreasing.
    diffsnorm(A, U, s, Va) outputs an estimate of the spectral norm of
    A - U diag(s) Va, which should be close to the machine precision.

    References
    ----------
    .. [PCA] Nathan Halko, Per-Gunnar Martinsson, and Joel Tropp,
             Finding structure with randomness: probabilistic
             algorithms for constructing approximate matrix
             decompositions, arXiv:0909.4061 [math.NA; math.PR], 2009
             (available at `arXiv <http://arxiv.org/abs/0909.4061>`_).

    See also
    --------
    diffsnorm, diffsnormc, eigens, eigenn
    """

    if l is None:
        l = k + 2

    (m, n) = A.shape

    assert k > 0
    assert k <= min(m, n)
    assert n_iter >= 0
    assert l >= k

    if np.isrealobj(A):
        isreal = True
    else:
        isreal = False

    # Promote the types of integer data to float data.
    dtype = (A * 1.0).dtype

    if raw:

        #
        # SVD A directly if l >= m/1.25 or l >= n/1.25.
        #
        if l >= m / 1.25 or l >= n / 1.25:
            (U, s, Va) = svd(A.todense() if issparse(A) else A, full_matrices=False)
            #
            # Retain only the leftmost k columns of U, the uppermost
            # k rows of Va, and the first k entries of s.
            #
            return U[:, :k], s[:k], Va[:k, :]

        if m >= n:

            #
            # Apply A to a random matrix, obtaining Q.
            #
            if isreal:
                Q = np.random.uniform(low=-1.0, high=1.0, size=(n, l)).astype(dtype)
                Q = mult(A, Q)
            if not isreal:
                Q = np.random.uniform(low=-1.0, high=1.0, size=(n, l)).astype(dtype)
                Q += 1j * np.random.uniform(low=-1.0, high=1.0, size=(n, l)).astype(
                    dtype
                )
                Q = mult(A, Q)

            #
            # Form a matrix Q whose columns constitute a
            # well-conditioned basis for the columns of the earlier Q.
            #
            if n_iter == 0:
                (Q, _) = qr(Q, mode="economic")
            if n_iter > 0:
                (Q, _) = lu(Q, permute_l=True)

            #
            # Conduct normalized power iterations.
            #
            for it in range(n_iter):

                Q = mult(Q.conj().T, A).conj().T

                (Q, _) = lu(Q, permute_l=True)

                Q = mult(A, Q)

                if it + 1 < n_iter:
                    (Q, _) = lu(Q, permute_l=True)
                else:
                    (Q, _) = qr(Q, mode="economic")

            #
            # SVD Q'*A to obtain approximations to the singular values
            # and right singular vectors of A; adjust the left singular
            # vectors of Q'*A to approximate the left singular vectors
            # of A.
            #
            QA = mult(Q.conj().T, A)
            (R, s, Va) = svd(QA, full_matrices=False)
            U = Q.dot(R)

            #
            # Retain only the leftmost k columns of U, the uppermost
            # k rows of Va, and the first k entries of s.
            #
            return U[:, :k], s[:k], Va[:k, :]

        if m < n:

            #
            # Apply A' to a random matrix, obtaining Q.
            #
            if isreal:
                R = np.random.uniform(low=-1.0, high=1.0, size=(l, m)).astype(dtype)
            if not isreal:
                R = np.random.uniform(low=-1.0, high=1.0, size=(l, m)).astype(dtype)
                R += 1j * np.random.uniform(low=-1.0, high=1.0, size=(l, m)).astype(
                    dtype
                )

            Q = mult(R, A).conj().T

            #
            # Form a matrix Q whose columns constitute a
            # well-conditioned basis for the columns of the earlier Q.
            #
            if n_iter == 0:
                (Q, _) = qr(Q, mode="economic")
            if n_iter > 0:
                (Q, _) = lu(Q, permute_l=True)

            #
            # Conduct normalized power iterations.
            #
            for it in range(n_iter):

                Q = mult(A, Q)
                (Q, _) = lu(Q, permute_l=True)

                Q = mult(Q.conj().T, A).conj().T

                if it + 1 < n_iter:
                    (Q, _) = lu(Q, permute_l=True)
                else:
                    (Q, _) = qr(Q, mode="economic")

            #
            # SVD A*Q to obtain approximations to the singular values
            # and left singular vectors of A; adjust the right singular
            # vectors of A*Q to approximate the right singular vectors
            # of A.
            #
            (U, s, Ra) = svd(mult(A, Q), full_matrices=False)
            Va = Ra.dot(Q.conj().T)

            #
            # Retain only the leftmost k columns of U, the uppermost
            # k rows of Va, and the first k entries of s.
            #
            return U[:, :k], s[:k], Va[:k, :]

    if not raw:

        #
        # Calculate the average of the entries in every column.
        #
        c = A.sum(axis=0) / m
        c = c.reshape((1, n))

        #
        # SVD the centered A directly if l >= m/1.25 or l >= n/1.25.
        #
        if l >= m / 1.25 or l >= n / 1.25:
            (U, s, Va) = svd(
                (A.todense() if issparse(A) else A)
                - np.ones((m, 1), dtype=dtype).dot(c),
                full_matrices=False,
            )
            #
            # Retain only the leftmost k columns of U, the uppermost
            # k rows of Va, and the first k entries of s.
            #
            return U[:, :k], s[:k], Va[:k, :]

        if m >= n:

            #
            # Apply the centered A to a random matrix, obtaining Q.
            #
            if isreal:
                R = np.random.uniform(low=-1.0, high=1.0, size=(n, l)).astype(dtype)
            if not isreal:
                R = np.random.uniform(low=-1.0, high=1.0, size=(n, l)).astype(dtype)
                R += 1j * np.random.uniform(low=-1.0, high=1.0, size=(n, l)).astype(
                    dtype
                )

            Q = mult(A, R) - np.ones((m, 1), dtype=dtype).dot(c.dot(R))

            #
            # Form a matrix Q whose columns constitute a
            # well-conditioned basis for the columns of the earlier Q.
            #
            if n_iter == 0:
                (Q, _) = qr(Q, mode="economic")
            if n_iter > 0:
                (Q, _) = lu(Q, permute_l=True)

            #
            # Conduct normalized power iterations.
            #
            for it in range(n_iter):

                Q = mult(Q.conj().T, A) - (
                    Q.conj().T.dot(np.ones((m, 1), dtype=dtype))
                ).dot(c)
                Q = Q.conj().T
                (Q, _) = lu(Q, permute_l=True)

                Q = mult(A, Q) - np.ones((m, 1), dtype=dtype).dot(c.dot(Q))

                if it + 1 < n_iter:
                    (Q, _) = lu(Q, permute_l=True)
                else:
                    (Q, _) = qr(Q, mode="economic")

            #
            # SVD Q' applied to the centered A to obtain
            # approximations to the singular values and right singular
            # vectors of the centered A; adjust the left singular
            # vectors to approximate the left singular vectors of the
            # centered A.
            #
            QA = mult(Q.conj().T, A) - (
                Q.conj().T.dot(np.ones((m, 1), dtype=dtype))
            ).dot(c)
            (R, s, Va) = svd(QA, full_matrices=False)
            U = Q.dot(R)

            #
            # Retain only the leftmost k columns of U, the uppermost
            # k rows of Va, and the first k entries of s.
            #
            return U[:, :k], s[:k], Va[:k, :]

        if m < n:

            #
            # Apply the adjoint of the centered A to a random matrix,
            # obtaining Q.
            #
            if isreal:
                R = np.random.uniform(low=-1.0, high=1.0, size=(l, m)).astype(dtype)
            if not isreal:
                R = np.random.uniform(low=-1.0, high=1.0, size=(l, m)).astype(dtype)
                R += 1j * np.random.uniform(low=-1.0, high=1.0, size=(l, m)).astype(
                    dtype
                )

            Q = mult(R, A) - (R.dot(np.ones((m, 1), dtype=dtype))).dot(c)
            Q = Q.conj().T

            #
            # Form a matrix Q whose columns constitute a
            # well-conditioned basis for the columns of the earlier Q.
            #
            if n_iter == 0:
                (Q, _) = qr(Q, mode="economic")
            if n_iter > 0:
                (Q, _) = lu(Q, permute_l=True)

            #
            # Conduct normalized power iterations.
            #
            for it in range(n_iter):

                Q = mult(A, Q) - np.ones((m, 1), dtype=dtype).dot(c.dot(Q))
                (Q, _) = lu(Q, permute_l=True)

                Q = mult(Q.conj().T, A) - (
                    Q.conj().T.dot(np.ones((m, 1), dtype=dtype))
                ).dot(c)
                Q = Q.conj().T

                if it + 1 < n_iter:
                    (Q, _) = lu(Q, permute_l=True)
                else:
                    (Q, _) = qr(Q, mode="economic")

            #
            # SVD the centered A applied to Q to obtain approximations
            # to the singular values and left singular vectors of the
            # centered A; adjust the right singular vectors to
            # approximate the right singular vectors of the centered A.
            #
            (U, s, Ra) = svd(
                mult(A, Q) - np.ones((m, 1), dtype=dtype).dot(c.dot(Q)),
                full_matrices=False,
            )
            Va = Ra.dot(Q.conj().T)

            #
            # Retain only the leftmost k columns of U, the uppermost
            # k rows of Va, and the first k entries of s.
            #
            return U[:, :k], s[:k], Va[:k, :]


def mult(A, B):
    """
    default matrix multiplication.

    Multiplies A and B together via the "dot" method.

    Parameters
    ----------
    A : array_like
        first matrix in the product A*B being calculated
    B : array_like
        second matrix in the product A*B being calculated

    Returns
    -------
    array_like
        product of the inputs A and B

    Examples
    --------
    >>> from fbpca import mult
    >>> from numpy import array
    >>> from numpy.linalg import norm
    >>>
    >>> A = array([[1., 2.], [3., 4.]])
    >>> B = array([[5., 6.], [7., 8.]])
    >>> norm(mult(A, B) - A.dot(B))

    This example multiplies two matrices two ways -- once with mult,
    and once with the usual "dot" method -- and then calculates the
    (Frobenius) norm of the difference (which should be near 0).
    """

    if issparse(B) and not issparse(A):
        # dense.dot(sparse) is not available in scipy.
        return B.T.dot(A.T).T
    else:
        return A.dot(B)


def set_matrix_mult(newmult):
    """
    re-definition of the matrix multiplication function "mult".

    Sets the matrix multiplication function "mult" used in fbpca to be
    the input "newmult" -- which must return the product A*B of its two
    inputs A and B, i.e., newmult(A, B) must be the product of A and B.

    Parameters
    ----------
    newmult : callable
        matrix multiplication replacing mult in fbpca; newmult must
        return the product of its two array_like inputs

    Returns
    -------
    None

    Examples
    --------
    >>> from fbpca import set_matrix_mult
    >>>
    >>> def newmult(A, B):
    ...     return A*B
    ...
    >>> set_matrix_mult(newmult)

    This example redefines the matrix multiplication used in fbpca to
    be the entrywise product.
    """

    global mult
    mult = newmult
