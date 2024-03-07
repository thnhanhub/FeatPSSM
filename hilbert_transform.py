import math

import numpy as np
from scipy.signal import hilbert2, hilbert


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# def spectr(anal_signal):
#     s = np.abs(anal_signal) ** 2
#     return np.array(s)

def extracted_features(anal_signal):
    features = []
    
    # 1. Năng lượng: Tổng bình phương của phần thực và phần ảo của biến đổi Hilbert
    energy = np.sum(np.abs(anal_signal) ** 2, axis=0)
    features.append(energy)

    # 2. Tần số trung bình: Tính toán trung bình của tần số của phần ảo của biến đổi Hilbert
    mean_frequency = np.mean(np.angle(anal_signal), axis=0)
    features.append(mean_frequency)

    # 3. Tần số cực đại: Tìm giá trị lớn nhất của tần số của phần ảo của biến đổi Hilbert
    max_frequency = np.max(np.angle(anal_signal), axis=0)
    features.append(max_frequency)

    # 4. Độ biến động: Phương sai của tần số của phần ảo của biến đổi Hilbert
    variance_frequency = np.var(np.angle(anal_signal), axis=0)
    features.append(variance_frequency)
    
    # 5. Tổng
    sum_l = np.sum(anal_signal.imag, axis=0)
    features.append(sum_l)
       
    return np.array(features)


def transform(X_, transformer=None):
    X_hat_ = []
    features = []
    for x in X_:
        # X_hat_.append(hilbert(x, N=40, axis=1).imag)
        # x = sigmoid(x)
        anal_sig = hilbert(x, N=40, axis=1)
        X_hat_.append(anal_sig.imag)
        features.append(extracted_features(anal_sig))
    stack = np.concatenate([X_hat_, features], 1)
    return np.array(stack)
