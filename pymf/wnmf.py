# Authors: Rikk Hill
# License: BSD 3 Clause
"""
PyMF Non-negative Matrix Factorization.

    WNMF: Class for Weighted Non-negative Matrix Factorization

[1] Lee, D. D. and Seung, H. S. (1999), Learning the Parts of Objects by Non-negative
Matrix Factorization, Nature 401(6755), 788-799.
"""
import numpy as np
from base import PyMFBase

__all__ = ["WNMF"]


class WNMF(PyMFBase):
    """
    WNMF(data, weights, num_bases=4)

    Non-negative Matrix Factorization. Factorize a data matrix into two matrices
    s.t. F = | S (*) (data - W*H) | is minimal. H, and W are restricted to non-negative
    data. S is a weighting matrix and (*) is Hadamard/elementwise multiplication.
    Uses the classicial multiplicative update rule.

    # (todo) Document this properly
    """
    def __init__(self, data, S, num_bases=4, **kwargs):
        PyMFBase.__init__(self, data, num_bases, **kwargs)
        self.S = S
        self.S_sqrt = np.sqrt(S)

    def _update_h(self):
        # pre init H1, and H2 (necessary for storing matrices on disk)
        H2 = np.dot(self.W.T, self.S_sqrt * np.dot(self.W, self.H)) + 10**-9
        self.H *= np.dot(self.W.T, self.S_sqrt * self.data[:, :])
        self.H /= H2

    def _update_w(self):
        # pre init W1, and W2 (necessary for storing matrices on disk)
        W2 = np.dot(self.S_sqrt * np.dot(self.W, self.H), self.H.T) + 10**-9
        self.W *= np.dot(self.S_sqrt * self.data[:, :], self.H.T)
        self.W /= W2
        self.W /= np.sqrt(np.sum(self.W**2.0, axis=0))


def _test():
    import doctest
    doctest.testmod()

if __name__ == "__main__":
    _test()