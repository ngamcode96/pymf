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
import scipy
from scipy import special

__all__ = ["PMF"]


class PMF(PyMFBase):
    """
        Poisson Matrix Factorisation
        Variational Bayesian factorisation
    """
    def __init__(self, data, num_bases=4, smoothness=100, **kwargs):

        data = data.T

        PyMFBase.__init__(self, data, num_bases, **kwargs)

        # Setup
        self.num_bases = num_bases
        self.smoothness = smoothness
        data_shape = data.shape
        self.w_height = data_shape[1]
        self.h_width = data_shape[0]

        self._parse_args(**kwargs)

    def _parse_args(self, **kwargs):
        self.a = float(kwargs.get('a', 0.1))
        self.b = float(kwargs.get('b', 0.1))

    def _init_w(self):
        self.gamma_w = self.smoothness \
            * np.random.gamma(self.smoothness, 1. / self.smoothness,
                              size=(self.num_bases, self.w_height))
        self.rho_w = self.smoothness \
            * np.random.gamma(self.smoothness, 1. / self.smoothness,
                              size=(self.num_bases, self.w_height))

        self.Ew, self.Elogw = _compute_expectations(self.gamma_w, self.rho_w)
        self.c = 1. / np.mean(self.Ew)

    def _init_h(self):
        self.gamma_h = self.smoothness \
            * np.random.gamma(self.smoothness, 1. / self.smoothness,
                              size=(self.h_width, self.num_bases))
        self.rho_h = self.smoothness \
            * np.random.gamma(self.smoothness, 1. / self.smoothness,
                              size=(self.h_width, self.num_bases))

        self.Eh, self.Elogh = _compute_expectations(self.gamma_h, self.rho_h)

    def _update_h(self):
        ratio = self.data / self._xexplog()
        self.gamma_h = self.a + np.exp(self.Elogh) * np.dot(
            ratio, np.exp(self.Elogw).T)
        self.rho_h = self.a * self.c + np.sum(self.Ew, axis=1)
        self.Eh, self.Elogh = _compute_expectations(self.gamma_h, self.rho_h)
        self.c = 1. / np.mean(self.Eh)

    def _update_w(self):
        ratio = self.data / self._xexplog()
        self.gamma_w = self.b + np.exp(self.Elogw) * np.dot(
            np.exp(self.Elogh).T, ratio)
        self.rho_w = self.b + np.sum(self.Eh, axis=0, keepdims=True).T
        self.Ew, self.Elogw = _compute_expectations(self.gamma_w, self.rho_w)

    def _xexplog(self):
        return np.dot(np.exp(self.Elogh), np.exp(self.Elogw))

    def frobenius_norm(self):
        """
        Not actually the Frobenius norm, but a comparable measure of the
        error bound for the approximation
        """
        error = np.sum(self.data * np.log(self._xexplog()) - self.Eh.dot(self.Ew))
        error += _gamma_term(self.a, self.a * self.c,
                             self.gamma_h, self.rho_h,
                             self.Eh, self.Elogh)
        error += self.num_bases * self.data.shape[0] * self.a * np.log(self.c)
        error += _gamma_term(self.b, self.b, self.gamma_w, self.rho_w,
                             self.Ew, self.Elogw)
        return error


def _compute_expectations(alpha, beta):
    '''
    Given x ~ Gam(alpha, beta), compute E[x] and E[log x]
    '''
    return alpha / beta, special.psi(alpha) - np.log(beta)


def _gamma_term(a, b, shape, rate, Ex, Elogx):
    return np.sum((a - shape) * Elogx - (b - rate) * Ex +
                  (special.gammaln(shape) - shape * np.log(rate)))


def _test():
    import doctest
    doctest.testmod()

if __name__ == "__main__":
    _test()