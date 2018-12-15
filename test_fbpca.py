"""
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
import logging
import unittest

import numpy as np
from scipy.linalg import eigh, qr, svd, norm
from scipy.sparse import coo_matrix, spdiags

from fbpca import diffsnorm, diffsnormc, diffsnorms, eigenn, eigens, pca


class TestDiffsnorm(unittest.TestCase):

    def test_dense(self):

        logging.info('running TestDiffsnorm.test_dense...')
        logging.info('err =')

        for dtype in ['float16', 'float32', 'float64']:
            if dtype == 'float64':
                prec = .1e-10
            elif dtype == 'float32':
                prec = .1e-2
            else:
                prec = .1e0
            for (m, n) in [(200, 100), (100, 200), (100, 100)]:
                for isreal in [True, False]:

                    if isreal:
                        A = np.random.normal(size=(m, n)).astype(dtype)
                    if not isreal:
                        A = np.random.normal(size=(m, n)).astype(dtype) \
                            + 1j * np.random.normal(size=(m, n)).astype(dtype)

                    (U, s, Va) = svd(A, full_matrices=False)
                    snorm = diffsnorm(A, U, s, Va)
                    logging.info(snorm)
                    self.assertTrue(snorm < prec * s[0])

    def test_sparse(self):

        logging.info('running TestDiffsnorm.test_sparse...')
        logging.info('err =')

        for dtype in ['float16', 'float32', 'float64']:
            if dtype == 'float64':
                prec = .1e-10
            elif dtype == 'float32':
                prec = .1e-2
            else:
                prec = .1e0
            for (m, n) in [(200, 100), (100, 200), (100, 100)]:
                for isreal in [True, False]:

                    if isreal:
                        A = 2 * spdiags(
                            np.arange(min(m, n)) + 1, 0, m, n).astype(dtype)
                    if not isreal:
                        A = 2 * spdiags(
                            np.arange(min(m, n)) + 1, 0, m, n).astype(dtype) \
                            * (1 + 1j)

                    A = A - spdiags(np.arange(min(m, n) + 1), 1, m, n)
                    A = A - spdiags(np.arange(min(m, n)) + 1, -1, m, n)
                    (U, s, Va) = svd(A.todense(), full_matrices=False)
                    A = A / s[0]

                    (U, s, Va) = svd(A.todense(), full_matrices=False)
                    snorm = diffsnorm(A, U, s, Va)
                    logging.info(snorm)
                    self.assertTrue(snorm < prec * s[0])


class TestDiffsnormc(unittest.TestCase):

    def test_dense(self):

        logging.info('running TestDiffsnormc.test_dense...')
        logging.info('err =')

        for dtype in ['float16', 'float32', 'float64']:
            if dtype == 'float64':
                prec = .1e-10
            elif dtype == 'float32':
                prec = .1e-2
            else:
                prec = .1e0
            for (m, n) in [(200, 100), (100, 200), (100, 100)]:
                for isreal in [True, False]:

                    if isreal:
                        A = np.random.normal(size=(m, n)).astype(dtype)
                    if not isreal:
                        A = np.random.normal(size=(m, n)).astype(dtype) \
                            + 1j * np.random.normal(size=(m, n)).astype(dtype)

                    c = A.sum(axis=0) / m
                    c = c.reshape((1, n))
                    Ac = A - np.ones((m, 1), dtype=dtype).dot(c)

                    (U, s, Va) = svd(Ac, full_matrices=False)
                    snorm = diffsnormc(A, U, s, Va)
                    logging.info(snorm)
                    self.assertTrue(snorm < prec * s[0])

    def test_sparse(self):

        logging.info('running TestDiffsnormc.test_sparse...')
        logging.info('err =')

        for dtype in ['float16', 'float32', 'float64']:
            if dtype == 'float64':
                prec = .1e-10
            elif dtype == 'float32':
                prec = .1e-2
            else:
                prec = .1e0
            for (m, n) in [(200, 100), (100, 200), (100, 100)]:
                for isreal in [True, False]:

                    if isreal:
                        A = 2 * spdiags(
                            np.arange(min(m, n)) + 1, 0, m, n).astype(dtype)
                    if not isreal:
                        A = 2 * spdiags(
                            np.arange(min(m, n)) + 1, 0, m, n).astype(dtype) \
                            * (1 + 1j)

                    A = A - spdiags(np.arange(min(m, n) + 1), 1, m, n)
                    A = A - spdiags(np.arange(min(m, n)) + 1, -1, m, n)
                    (U, s, Va) = svd(A.todense(), full_matrices=False)
                    A = A / s[0]

                    Ac = A.todense()
                    c = Ac.sum(axis=0) / m
                    c = c.reshape((1, n))
                    Ac = Ac - np.ones((m, 1), dtype=dtype).dot(c)

                    (U, s, Va) = svd(Ac, full_matrices=False)
                    snorm = diffsnormc(A, U, s, Va)
                    logging.info(snorm)
                    self.assertTrue(snorm < prec * s[0])


class TestDiffsnorms(unittest.TestCase):

    def test_dense(self):

        logging.info('running TestDiffsnorms.test_dense...')
        logging.info('err =')

        for dtype in ['float16', 'float32', 'float64']:
            if dtype == 'float64':
                prec = .1e-10
            elif dtype == 'float32':
                prec = .1e-2
            else:
                prec = .1e0
            for n in [100, 200]:
                for isreal in [True, False]:

                    if isreal:
                        A = np.random.normal(size=(n, n)).astype(dtype)
                    if not isreal:
                        A = np.random.normal(size=(n, n)).astype(dtype) \
                            + 1j * np.random.normal(size=(n, n)).astype(dtype)

                    (U, s, Va) = svd(A, full_matrices=True)
                    T = (np.diag(s).dot(Va)).dot(U)
                    snorm = diffsnorms(A, T, U)
                    logging.info(snorm)
                    self.assertTrue(snorm < prec * s[0])

    def test_sparse(self):

        logging.info('running TestDiffsnorms.test_sparse...')
        logging.info('err =')

        for dtype in ['float16', 'float32', 'float64']:
            if dtype == 'float64':
                prec = .1e-10
            elif dtype == 'float32':
                prec = .1e-2
            else:
                prec = .1e0
            for n in [100, 200]:
                for isreal in [True, False]:

                    if isreal:
                        A = 2 * spdiags(
                            np.arange(n) + 1, 0, n, n).astype(dtype)
                    if not isreal:
                        A = 2 * spdiags(
                            np.arange(n) + 1, 0, n, n).astype(dtype) * (1 + 1j)

                    A = A - spdiags(np.arange(n + 1), 1, n, n)
                    A = A - spdiags(np.arange(n) + 1, -1, n, n)
                    (U, s, Va) = svd(A.todense(), full_matrices=False)
                    A = A / s[0]
                    A = A.tocoo()

                    (U, s, Va) = svd(A.todense(), full_matrices=True)
                    T = (np.diag(s).dot(Va)).dot(U)
                    snorm = diffsnorms(A, T, U)
                    logging.info(snorm)
                    self.assertTrue(snorm < prec * s[0])


class TestEigenn(unittest.TestCase):

    def test_dense(self):

        logging.info('running TestEigenn.test_dense...')

        errs = []
        err = []

        def eigenntesterrs(n, k, n_iter, isreal, l, dtype):

            if isreal:
                V = np.random.normal(size=(n, k)).astype(dtype)
            if not isreal:
                V = np.random.normal(size=(n, k)).astype(dtype) \
                    + 1j * np.random.normal(size=(n, k)).astype(dtype)

            (V, _) = qr(V, mode='economic')

            d0 = np.zeros((k), dtype=dtype)
            d0[0] = 1
            d0[1] = .1
            d0[2] = .01

            A = V.dot(np.diag(d0).dot(V.conj().T))
            A = (A + A.conj().T) / 2

            (d1, V1) = eigh(A)
            (d2, V2) = eigenn(A, k, n_iter, l)

            d3 = np.zeros((n), dtype=dtype)
            for ii in range(k):
                d3[ii] = d2[ii]
            d3 = sorted(d3)
            errsa = norm(d1 - d3)

            erra = diffsnorms(A, np.diag(d2), V2)

            return erra, errsa

        for dtype in ['float16', 'float32', 'float64']:
            if dtype == 'float64':
                prec = .1e-10
            elif dtype == 'float32':
                prec = .1e-2
            else:
                prec = .1e0
            for n in [10, 20]:
                for k in [3, 9]:
                    for n_iter in [0, 2, 1000]:
                        for isreal in [True, False]:
                            l = k + 1
                            (erra, errsa) = eigenntesterrs(
                                n, k, n_iter, isreal, l, dtype)
                            err.append(erra)
                            errs.append(errsa)
                            self.assertTrue(erra < prec)

        logging.info('errs = \n%s', np.asarray(errs))
        logging.info('err = \n%s', np.asarray(err))

    def test_sparse(self):

        logging.info('running TestEigenn.test_sparse...')

        errs = []
        err = []
        bests = []

        def eigenntestserrs(n, k, n_iter, isreal, l, dtype):

            n2 = int(round(n / 2))
            assert 2 * n2 == n

            A = 2 * spdiags(np.arange(n2) + 1, 0, n, n).astype(dtype)

            if isreal:
                A = A - spdiags(np.arange(n2 + 1), 1, n, n).astype(dtype)
                A = A - spdiags(np.arange(n2) + 1, -1, n, n).astype(dtype)

            if not isreal:
                A = A - 1j * spdiags(np.arange(n2 + 1), 1, n, n).astype(dtype)
                A = A + 1j * spdiags(np.arange(n2) + 1, -1, n, n).astype(dtype)

            A = A / diffsnorms(
                A, np.zeros((2, 2), dtype=dtype),
                np.zeros((n, 2), dtype=dtype))

            A = A.dot(A)
            A = A.dot(A)
            A = A.dot(A)
            A = A.dot(A)

            A = A.tocoo()

            P = np.random.permutation(n)
            A = coo_matrix((A.data, (P[A.row], P[A.col])), shape=(n, n))
            A = A.tocsr()

            (d1, V1) = eigh(A.toarray())
            (d2, V2) = eigenn(A, k, n_iter, l)

            bestsa = sorted(abs(d1))[-k - 1]

            d3 = np.zeros((n), dtype=dtype)
            for ii in range(k):
                d3[ii] = d2[ii]
            d3 = sorted(d3)
            errsa = norm(d1 - d3)

            erra = diffsnorms(A, np.diag(d2), V2)

            return erra, errsa, bestsa

        for dtype in ['float16', 'float32', 'float64']:
            if dtype == 'float64':
                prec = .1e-10
            elif dtype == 'float32':
                prec = .1e-2
            else:
                prec = .1e0
            for n in [100, 200]:
                for k in [30, 90]:
                    for n_iter in [2, 1000]:
                        for isreal in [True, False]:
                            l = k + 1
                            (erra, errsa, bestsa) = eigenntestserrs(
                                n, k, n_iter, isreal, l, dtype)
                            err.append(erra)
                            errs.append(errsa)
                            bests.append(bestsa)
                            self.assertTrue(erra < max(10 * bestsa, prec))

        logging.info('errs = \n%s', np.asarray(errs))
        logging.info('err = \n%s', np.asarray(err))
        logging.info('bests = \n%s', np.asarray(bests))


class TestEigens(unittest.TestCase):

    def test_dense(self):

        logging.info('running TestEigens.test_dense...')

        errs = []
        err = []

        def eigenstesterrs(n, k, n_iter, isreal, l, dtype):

            if isreal:
                V = np.random.normal(size=(n, k)).astype(dtype)
            if not isreal:
                V = np.random.normal(size=(n, k)).astype(dtype) \
                    + 1j * np.random.normal(size=(n, k)).astype(dtype)

            (V, _) = qr(V, mode='economic')

            d0 = np.zeros((k), dtype=dtype)
            d0[0] = 1
            d0[1] = -.1
            d0[2] = .01

            A = V.dot(np.diag(d0).dot(V.conj().T))
            A = (A + A.conj().T) / 2

            (d1, V1) = eigh(A)
            (d2, V2) = eigens(A, k, n_iter, l)

            d3 = np.zeros((n), dtype=dtype)
            for ii in range(k):
                d3[ii] = d2[ii]
            d3 = sorted(d3)
            errsa = norm(d1 - d3)

            erra = diffsnorms(A, np.diag(d2), V2)

            return erra, errsa

        for dtype in ['float64', 'float32', 'float16']:
            if dtype == 'float64':
                prec = .1e-10
            elif dtype == 'float32':
                prec = .1e-2
            else:
                prec = .1e0
            for n in [10, 20]:
                for k in [3, 9]:
                    for n_iter in [0, 2, 1000]:
                        for isreal in [True, False]:
                            l = k + 1
                            (erra, errsa) = eigenstesterrs(
                                n, k, n_iter, isreal, l, dtype)
                            err.append(erra)
                            errs.append(errsa)
                            self.assertTrue(erra < prec)

        logging.info('errs = \n%s', np.asarray(errs))
        logging.info('err = \n%s', np.asarray(err))

    def test_sparse(self):

        logging.info('running TestEigens.test_sparse...')

        errs = []
        err = []
        bests = []

        def eigenstestserrs(n, k, n_iter, isreal, l, dtype):

            n2 = int(round(n / 2))
            assert 2 * n2 == n

            A = 2 * spdiags(np.arange(n2) + 1, 0, n2, n2).astype(dtype)

            if isreal:
                A = A - spdiags(np.arange(n2 + 1), 1, n2, n2).astype(dtype)
                A = A - spdiags(np.arange(n2) + 1, -1, n2, n2).astype(dtype)

            if not isreal:
                A = A - 1j * spdiags(
                    np.arange(n2 + 1), 1, n2, n2).astype(dtype)
                A = A + 1j * spdiags(
                    np.arange(n2) + 1, -1, n2, n2).astype(dtype)

            A = A / diffsnorms(
                A, np.zeros((2, 2), dtype=dtype),
                np.zeros((n2, 2), dtype=dtype))

            A = A.dot(A)
            A = A.dot(A)
            A = A.dot(A)
            A = A.dot(A)

            A = A.tocoo()

            datae = np.concatenate([A.data, A.data])
            rowe = np.concatenate([A.row + n2, A.row])
            cole = np.concatenate([A.col, A.col + n2])
            A = coo_matrix((datae, (rowe, cole)), shape=(n, n))

            P = np.random.permutation(n)
            A = coo_matrix((A.data, (P[A.row], P[A.col])), shape=(n, n))
            A = A.tocsc()

            (d1, V1) = eigh(A.toarray())
            (d2, V2) = eigens(A, k, n_iter, l)

            bestsa = sorted(abs(d1))[-k - 1]

            d3 = np.zeros((n), dtype=dtype)
            for ii in range(k):
                d3[ii] = d2[ii]
            d3 = sorted(d3)
            errsa = norm(d1 - d3)

            erra = diffsnorms(A, np.diag(d2), V2)

            return erra, errsa, bestsa

        for dtype in ['float16', 'float32', 'float64']:
            if dtype == 'float64':
                prec = .1e-10
            elif dtype == 'float32':
                prec = .1e-2
            else:
                prec = .1e0
            for n in [100, 200]:
                for k in [30, 90]:
                    for n_iter in [2, 1000]:
                        for isreal in [True, False]:
                            l = k + 1
                            (erra, errsa, bestsa) = eigenstestserrs(
                                n, k, n_iter, isreal, l, dtype)
                            err.append(erra)
                            errs.append(errsa)
                            bests.append(bestsa)
                            self.assertTrue(erra < max(10 * bestsa, prec))

        logging.info('errs = \n%s', np.asarray(errs))
        logging.info('err = \n%s', np.asarray(err))
        logging.info('bests = \n%s', np.asarray(bests))


class TestPCA(unittest.TestCase):

    def test_dense(self):

        logging.info('running TestPCA.test_dense...')

        errs = []
        err = []

        def pcatesterrs(m, n, k, n_iter, raw, isreal, l, dtype):

            if isreal:
                U = np.random.normal(size=(m, k)).astype(dtype)
                (U, _) = qr(U, mode='economic')

                V = np.random.normal(size=(n, k)).astype(dtype)
                (V, _) = qr(V, mode='economic')

            if not isreal:
                U = np.random.normal(size=(m, k)).astype(dtype) \
                    + 1j * np.random.normal(size=(m, k)).astype(dtype)
                (U, _) = qr(U, mode='economic')

                V = np.random.normal(size=(n, k)).astype(dtype) \
                    + 1j * np.random.normal(size=(n, k)).astype(dtype)
                (V, _) = qr(V, mode='economic')

            s0 = np.zeros((k), dtype=dtype)
            s0[0] = 1
            s0[1] = .1
            s0[2] = .01

            A = U.dot(np.diag(s0).dot(V.conj().T))

            if raw:
                Ac = A
            if not raw:
                c = A.sum(axis=0) / m
                c = c.reshape((1, n))
                Ac = A - np.ones((m, 1), dtype=dtype).dot(c)

            (U, s1, Va) = svd(Ac, full_matrices=False)
            (U, s2, Va) = pca(A, k, raw, n_iter, l)

            s3 = np.zeros((min(m, n)), dtype=dtype)
            for ii in range(k):
                s3[ii] = s2[ii]
            errsa = norm(s1 - s3)

            if raw:
                erra = diffsnorm(A, U, s2, Va)
            if not raw:
                erra = diffsnormc(A, U, s2, Va)

            return erra, errsa

        for dtype in ['float64', 'float32', 'float16']:
            if dtype == 'float64':
                prec = .1e-10
            elif dtype == 'float32':
                prec = .1e-2
            else:
                prec = .1e0
            for (m, n) in [(20, 10), (10, 20), (20, 20)]:
                for k in [3, 9]:
                    for n_iter in [0, 2, 1000]:
                        for raw in [True, False]:
                            for isreal in [True, False]:
                                l = k + 1
                                (erra, errsa) = pcatesterrs(
                                    m, n, k, n_iter, raw, isreal, l, dtype)
                                err.append(erra)
                                errs.append(errsa)
                                self.assertTrue(erra < prec)

        logging.info('errs = \n%s', np.asarray(errs))
        logging.info('err = \n%s', np.asarray(err))

    def test_sparse(self):

        logging.info('running TestPCA.test_sparse...')

        errs = []
        err = []
        bests = []

        def pcatestserrs(m, n, k, n_iter, raw, isreal, l, dtype):

            if isreal:
                A = 2 * spdiags(np.arange(min(m, n)) + 1, 0, m, n)
            if not isreal:
                A = 2 * spdiags(np.arange(min(m, n)) + 1, 0, m, n)
                A = A.astype(dtype) * (1 + 1j)

            A = A - spdiags(np.arange(min(m, n) + 1), 1, m, n)
            A = A - spdiags(np.arange(min(m, n)) + 1, -1, m, n)
            A = A / diffsnorm(
                A, np.zeros((m, 2), dtype=dtype), [0, 0],
                np.zeros((2, n), dtype=dtype))
            A = A.dot(A.conj().T.dot(A))
            A = A.dot(A.conj().T.dot(A))
            A = A[np.random.permutation(m), :]
            A = A[:, np.random.permutation(n)]

            if raw:
                Ac = A
            if not raw:
                c = A.sum(axis=0) / m
                c = c.reshape((1, n))
                Ac = A - np.ones((m, 1), dtype=dtype).dot(c)

            if raw:
                (U, s1, Va) = svd(Ac.toarray(), full_matrices=False)
            if not raw:
                (U, s1, Va) = svd(Ac, full_matrices=False)

            (U, s2, Va) = pca(A, k, raw, n_iter, l)

            bestsa = s1[k]

            s3 = np.zeros((min(m, n)), dtype=dtype)
            for ii in range(k):
                s3[ii] = s2[ii]
            errsa = norm(s1 - s3)

            if raw:
                erra = diffsnorm(A, U, s2, Va)
            if not raw:
                erra = diffsnormc(A, U, s2, Va)

            return erra, errsa, bestsa

        for dtype in ['float64', 'float32', 'float16']:
            if dtype == 'float64':
                prec = .1e-10
            elif dtype == 'float32':
                prec = .1e-2
            else:
                prec = .1e0
            for (m, n) in [(200, 100), (100, 200), (100, 100)]:
                for k in [30, 90]:
                    for n_iter in [2, 1000]:
                        for raw in [True, False]:
                            for isreal in [True, False]:
                                l = k + 1
                                (erra, errsa, bestsa) = pcatestserrs(
                                    m, n, k, n_iter, raw, isreal, l, dtype)
                                err.append(erra)
                                errs.append(errsa)
                                bests.append(bestsa)
                                self.assertTrue(erra < max(10 * bestsa, prec))

        logging.info('errs = \n%s', np.asarray(errs))
        logging.info('err = \n%s', np.asarray(err))
        logging.info('bests = \n%s', np.asarray(bests))


if __name__ == '__main__':
    logging.basicConfig(format='%(message)s', level=logging.INFO)
    unittest.main()
