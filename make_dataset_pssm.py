""" code: thnhan

USAGE:

Sử dụng code này để lấy PSSM từ file PSSM.
PSSM sau khi được lấy, sẽ được vào file pkl.
PSSM là ma trận có kích thước L*20, L là chiều dài chuỗi.

Chỉ cần `list_pssm = pickle.load(path_to_file)`,
thì `list_pssm` chứa danh sách các PSSM[L*20].
CHÚ Ý: L thay đổi.
"""

import glob
import pickle

import numpy as np
import pandas as pd
from zipfile import ZipFile


# def load_lst_file(file_lst, dir_name):
#     lst_file = load_text_file(file_lst)
#     lst_path = [dir_name + file_name.strip() for file_name in lst_file]
#     return lst_path
#
#
# def create_list_train():
#     lst_path_positive_train = glob.glob("./TrainDataPSSM/positive/*.pssm")
#     lst_path_negative_train = glob.glob("./TrainDataPSSM/negative/*.pssm")
#
#     print("Positive train: ", len(lst_path_positive_train))
#     print("Negative train: ", len(lst_path_negative_train))
#
#     lst_positive_train_label = [1] * len(lst_path_positive_train)
#     lst_negative_train_label = [0] * len(lst_path_negative_train)
#
#     lst_path_train = lst_path_positive_train + lst_path_negative_train
#     lst_label_train = lst_positive_train_label + lst_negative_train_label
#
#     print("Train all: ", len(lst_path_train))
#     print("")
#     return lst_path_train, lst_label_train


def get_PSSM(zip_file_name, file_name, n_cols_take=20):
    with ZipFile(zip_file_name) as z:
        with z.open(file_name) as f:
            readf = f.readlines()
            lines = [line.decode("utf-8") for line in readf]
            start_line = 3

            end_line = 0
            while lines[end_line].find(r"Lambda") == -1:
                end_line += 1
            end_line -= 2
            # print(end_line)
            # print(lines[end_line])
            # print(end_line - start_line + 1)
            values = np.zeros((end_line - start_line + 1, n_cols_take))

            for i in range(start_line, end_line + 1):
                strs = lines[i].strip().split()[2:22]
                for j in range(20):
                    values[i - start_line][j] = int(strs[j])
        return values


def make_dataset(path_to_datset_dir):
    def make_from_pairs(df_pairs: pd.DataFrame, prefix_save):
        list_pssm_A = []
        for prot_id in df_pairs['proteinA']:
            list_pssm_A.append(get_PSSM(path_to_datset_dir + "/PSSM_profile.zip", prot_id + ".pssm"))
        pickle.dump(list_pssm_A, open(prefix_save + "_A_pssm.pkl", "wb"))
        print("Number of PSSMs;", len(list_pssm_A))
        print("Size of the first PSSM;", list_pssm_A[0].shape)
        print("Saved")

        list_pssm_B = []
        for prot_id in df_pairs['proteinB']:
            list_pssm_B.append(get_PSSM(path_to_datset_dir + "/PSSM_profile.zip", prot_id + ".pssm"))
        pickle.dump(list_pssm_B, open(prefix_save + "_B_pssm.pkl", "wb"))
        print("Number of PSSMs;", len(list_pssm_B))
        print("Size of the first PSSM;", list_pssm_B[0].shape)
        print("Saved")

        return 0

    print("\nGet PSSM ...")

    pos_pairs = pd.read_csv(path_to_datset_dir + '/positive.txt', sep='\t')
    make_from_pairs(pos_pairs, 'pos')
    neg_pairs = pd.read_csv(path_to_datset_dir + '/negative.txt', sep='\t')
    make_from_pairs(neg_pairs, 'neg')

    return 0


if __name__ == "__main__":
    make_dataset("dataset/YeastCore")
