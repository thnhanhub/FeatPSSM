import math

import numpy as np
from scipy.signal import hilbert2, hilbert
from plot_hilbert_pssm import plot_h_p


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def transform(X_, transformer=None):
    X_hat_ = []
    for x in X_:
        # X_hat_.append(hilbert(x, N=40, axis=1).real)
        # x = sigmoid(x)
        X_hat_.append(hilbert(x).imag)
    return np.array(X_hat_)
