import pickle

import numpy as np
import pandas as pd

def get_PSSM(file_name):
    with open(file_name) as f:
        lines = f.readlines()
        start_line = 3

        end_line = 0
        while lines[end_line].find(r"Lambda") == -1:
            end_line += 1
        end_line -= 2
        # print(end_line)
        # print(lines[end_line])
        # print(end_line - start_line + 1)
        values = np.zeros((end_line - start_line + 1, 20))

        for i in range(start_line, end_line + 1):
            strs = lines[i].strip().split()[2:22]
            for j in range(20):
                values[i - start_line][j] = int(strs[j])
    return values


if __name__ == "__main__":
    get_PSSM(<ten file pssm>)
