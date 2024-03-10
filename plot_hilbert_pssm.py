import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert


def plot_h_p(lst_pssm, len):
        lst_hilbert = hilbert(lst_pssm[:len]).imag
        
        plt.figure(figsize=(12, 6))
        plt.plot(lst_pssm[:len].flatten(), marker='o', markersize=5, label="pssm")        
        plt.plot(lst_hilbert.flatten(), marker='o', markersize=5, label="hilbert")
        plt.title("Đồ thị biểu diễn ma trận PSSM và sau khi áp dụng biến đổi Hilbert")
        plt.xlabel("Vị trí")
        plt.ylabel("Điểm")
        plt.legend()
        plt.show()

if __name__ == "__main__":
        pssm_matrix = np.array([[3, 6, 1, 0, 0, 6, 7, 2, 1],
                [2, 2, 1, 0, 0, 2, 1, 1, 2],
                [1, 3, 7, 10, 5, 3, 2, 1, 4], 
                [1, 4, 7, 2, 3, 6, 2, 5, 3]])

        hilbert_matrix = hilbert(pssm_matrix).imag
        
        plot_h_p(pssm_matrix)


