import math

import numpy as np
from scipy.signal import hilbert2, hilbert


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def extracted_features(anal_signal):
    features = []
    magnitude = np.abs(anal_signal)
    phase = np.angle(anal_signal)

    # Độ lệch chuẩn:
    std_magnitude = np.std(magnitude, axis=0)
    features.append(std_magnitude)
    
    # Năng lượng: Tổng bình phương của phần thực và phần ảo của biến đổi Hilbert
    energy = np.sum(magnitude ** 2, axis=0)
    features.append(energy)

    # Tần số trung bình: Tính toán trung bình của tần số của phần ảo của biến đổi Hilbert
    mean_frequency = np.mean(phase, axis=0)
    features.append(mean_frequency)

    # Tần số cực đại: Tìm giá trị lớn nhất của tần số của phần ảo của biến đổi Hilbert
    max_frequency = np.max(phase, axis=0)
    features.append(max_frequency)
    
    # Độ biến động: Phương sai của tần số của phần ảo của biến đổi Hilbert
    variance_frequency = np.var(phase, axis=0)
    features.append(variance_frequency)

    return np.array(features)


def transform(X_, transformer=None):
    X_hat_ = []
    features = []
    for x in X_:
        # X_hat_.append(hilbert(x, N=40, axis=1).imag)
        # x = sigmoid(x)
        anal_signal = hilbert(x, axis=1)
        X_hat_.append(anal_signal.imag)
        # X_hat_.append([np.sum(anal_signal.imag, axis=0), np.sum(x, axis=0)])
        features.append(extracted_features(anal_signal))
    # stack = np.concatenate([X_hat_, features], axis=1)
    return np.array(features), np.array(X_hat_)
