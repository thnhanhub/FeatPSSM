import math

import numpy as np
from scipy.signal import hilbert2, hilbert
from scipy.fft import fft2

def sigmoid(x):
    return 1 / (1 + np.exp(-x))
    
def extracted_features(analytic_signal):
    features = []
    magnitude = np.abs(analytic_signal)
    phase = np.angle(analytic_signal)
            
    # Biên độ
    energy = np.sum(magnitude ** 2, axis=0)
    features.extend(energy)
    
    mean_magnitude = np.mean(magnitude, axis=0)
    features.extend(mean_magnitude)
    
    mean_phase = np.mean(phase, axis=0)
    features.extend(mean_phase)

    max_magnitude = np.max(magnitude, axis=0)
    features.extend(max_magnitude)
    
    max_phase = np.max(phase, axis=0)
    features.extend(max_phase)
    
    min_magnitude = np.min(magnitude, axis=0)
    features.extend(min_magnitude)

    min_phase = np.min(phase, axis=0)
    features.extend(min_phase)

    variance_magnitude = np.var(magnitude, axis=0)    
    features.extend(variance_magnitude)

    variance_phase = np.var(phase, axis=0)
    features.extend(variance_phase)

    std_magnitude = np.std(magnitude, axis=0)
    features.extend(std_magnitude)
    
    std_phase = np.std(phase, axis=0)
    features.extend(std_phase)
    
    max_edge_magnitude = np.argmax(magnitude, axis=0)
    features.extend(max_edge_magnitude)
    
    max_edge_phase = np.argmax(phase, axis=0)
    features.extend(max_edge_phase)

    return np.array(features)


def transform(X_, transformer=None):
    X_hat_ = []
    features = []
    for x in X_:
        # X_hat_.append(hilbert(x, N=40, axis=1).imag)
        # x = sigmoid(x)
        anal_signal = hilbert(x, axis=1)
        X_hat_.append(np.sum(anal_signal.imag, axis=0))
        features.append(extracted_features(anal_signal))
    features = np.concatenate([features, X_hat_], axis=1)
    return features
