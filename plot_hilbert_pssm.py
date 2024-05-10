import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert


def plot_h_p(lst_pssm, len):
        fix_pssm = lst_pssm[:len].reshape(-1)
        lst_hilbert = hilbert(fix_pssm).imag

        # lst_hilbert = hilbert(lst_pssm)
        mean_ = np.mean([fix_pssm, lst_hilbert], axis=0)
        std_ = np.std([fix_pssm, lst_hilbert], axis=0)

        # std_ampl = np.std(amp, axis=0)
        plt.rc("font", size=14)
        plt.figure(figsize=(14, 7))
        plt.plot(mean_, label="Trung bình")        
        plt.plot(std_, label="Độ lệch chuẩn")
        plt.title("Đồ thị biểu diễn ma trận PSSM và sau khi áp dụng biến đổi Hilbert.")
        plt.xlabel("Vị trí")
        plt.ylabel("Điểm")
        plt.legend()
        plt.show()

if __name__ == "__main__":
        pssm_matrix = np.array([[3, 6, 1, 0, 0, 6, 7, 2, 1],
                [2, 2, 1, 0, 0, 2, 1, 1, 2],
                [1, 3, 7, 10, 5, 3, 2, 1, 4], 
                [1, 4, 7, 2, 3, 6, 2, 5, 3]])
        
        plot_h_p(pssm_matrix, 4)


