import numpy as np
import pandas as pd
from scipy.io import loadmat, savemat


def read_mat_to_feat(file_name):
    """
    Usage:  `prot_feat = read_mat_to_feat("file_name.mat")`
    """
    data = loadmat(file_name)
    # print(data)
    prot_id = np.array([i[0][0] for i in data['prot_id']])
    features = data['prot_feat']
    df_ = pd.DataFrame(data=features, index=prot_id)
    # print(protID)
    # print(features)
    return df_


def save_feat_to_mat(file_name, data):
    a = np.array([np.array(['prot_2528'], dtype='<U9'), np.array(['prot_2528'], dtype='<U9')], dtype='object')
    b = np.array([['prot_2528'], ['prot_2528']], dtype='object')
    print(a)
    savemat('hehe.mat', {'tam': a, 'tam2': b})
    print(loadmat('hehe.mat'))





if __name__ == "__main__":
    prot_feat = read_mat_to_feat("../uni_Fvector.mat")
    print(prot_feat)
