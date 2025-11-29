"""Utilities for estimating hyperspectral noise statistics and subspace size."""

import numpy as np
import scipy 
from spectral import*
import os
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.svm import SVC
from pysptools.material_count import vd as v
from sklearn.preprocessing import MinMaxScaler

def est_noise(y, noise_type='additive'):
    """
    Infer hyperspectral noise by regressing each band against all others.

    Parameters
    ----------
    y : numpy.ndarray
        Hyperspectral cube reshaped as ``((m*n) x p)``.
    noise_type : str, optional
        Either ``'additive'`` or ``'poisson'`` depending on the assumed noise.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        Estimated noise samples for every pixel and the corresponding
        correlation matrix.
    """
    def est_additive_noise(r):
        """
        Estimate additive noise statistics via leave-one-band-out regression.

        Parameters
        ----------
        r : numpy.ndarray
            Matrix of shape ``(bands, pixels)`` representing spectra.

        Returns
        -------
        tuple[numpy.ndarray, numpy.ndarray]
            Estimated noise samples and their covariance matrix.
        """
        small = 1e-6
        L, N = r.shape
        w=np.zeros((L,N), dtype=float)
        RR=np.dot(r,r.T)
        RRi = np.linalg.pinv(RR+small*np.eye(L))
        RRi = np.matrix(RRi)
        for i in range(L):
            XX = RRi - (RRi[:,i]*RRi[i,:]) / RRi[i,i]
            RRa = RR[:,i]
            RRa[i] = 0
            beta = np.dot(XX, RRa)
            beta[0,i]=0
            w[i,:] = r[i,:] - np.dot(beta,r)
        Rw = np.diag(np.diag(np.dot(w,w.T) / N))
        return w, Rw

    y = y.T
    L, N = y.shape
    if noise_type == 'poisson':
        sqy = np.sqrt(y * (y > 0))
        u, Ru = est_additive_noise(sqy)
        x = (sqy - u)**2
        w = np.sqrt(x)*u*2
        Rw = np.dot(w,w.T) / N
    else:
        w, Rw = est_additive_noise(y)
    return w.T, Rw.T


def hysime(y, n, Rn):
    """
    Estimate the signal subspace dimension using the HySime algorithm.

    Parameters
    ----------
    y : numpy.ndarray
        Hyperspectral data set with shape ``((m*n) x p)``.
    n : numpy.ndarray
        Estimated noise with the same shape as ``y``.
    Rn : numpy.ndarray
        Noise correlation matrix with shape ``(p x p)``.

    Returns
    -------
    tuple[int, numpy.ndarray, numpy.ndarray, numpy.ndarray]
        Estimated subspace dimension, the eigenvectors spanning the subspace,
        the full eigenvector matrix, and the HySime cost for each component.
    """
    y=y.T
    n=n.T
    Rn=Rn.T
    L, N = y.shape
    Ln, Nn = n.shape
    d1, d2 = Rn.shape

    x = y - n

    Ry = np.dot(y, y.T) / N
    Rx = np.dot(x, x.T) / N
    E, dx, V = np.linalg.svd(Rx)

    Rn = Rn+np.sum(np.diag(Rx))/L/10**5 * np.eye(L)
    Py = np.diag(np.dot(E.T, np.dot(Ry,E)))
    Pn = np.diag(np.dot(E.T, np.dot(Rn,E)))
    cost_F = Py - 2 * Pn
    kf = np.sum(cost_F > 0)
    ind_asc = np.argsort(cost_F)
    Ek = E[:, ind_asc[0:kf]]
    return kf, Ek,E,cost_F
