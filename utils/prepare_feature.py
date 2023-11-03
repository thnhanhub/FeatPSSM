import numpy as np
import pandas as pd
from scipy.io import loadmat
from sklearn.decomposition import PCA

from utils.read_mat_file import read_mat_to_feat


def load_positive(dset_name, dset_dir='data', feat_dir='extracted_feature'):
    pos = pd.read_csv(dset_dir + '/' + dset_name + '/pairs_pos.csv')

    Fvector = read_mat_to_feat(feat_dir + '/' + dset_name + '/uni_Fvector.mat')
    pos_Fvector = np.concatenate([Fvector.loc[pos.proteinA], Fvector.loc[pos.proteinB]], axis=1)

    LD = read_mat_to_feat(feat_dir + '/' + dset_name + '/uni_LD.mat')
    pos_LD = np.concatenate([LD.loc[pos.proteinA], LD.loc[pos.proteinB]], axis=1)

    APAAC = read_mat_to_feat(feat_dir + '/' + dset_name + '/uni_APAAC.mat')
    pos_APAAC = np.concatenate([APAAC.loc[pos.proteinA], APAAC.loc[pos.proteinB]], axis=1)

    pos_feat = np.concatenate([pos_Fvector, pos_LD, pos_APAAC], axis=1)

    return pos_feat


def load_negative(dset_name, dset_dir='data', feat_dir='extracted_feature'):
    neg = pd.read_csv(dset_dir + '/' + dset_name + '/pairs_neg.csv')

    Fvector = read_mat_to_feat(feat_dir + '/' + dset_name + '/uni_Fvector.mat')
    neg_Fvector = np.concatenate([Fvector.loc[neg.proteinA], Fvector.loc[neg.proteinB]], axis=1)

    LD = read_mat_to_feat(feat_dir + '/' + dset_name + '/uni_LD.mat')
    neg_LD = np.concatenate([LD.loc[neg.proteinA], LD.loc[neg.proteinB]], axis=1)

    APAAC = read_mat_to_feat(feat_dir + '/' + dset_name + '/uni_APAAC.mat')
    neg_APAAC = np.concatenate([APAAC.loc[neg.proteinA], APAAC.loc[neg.proteinB]], axis=1)

    neg_feat = np.concatenate([neg_Fvector, neg_LD, neg_APAAC], axis=1)

    return neg_feat


def load_feature_lable(dset_name, dset_dir='data', feat_dir='extracted_feature', only_posivite=0):
    """
        dset_name: vd: Yeast, Human,
    Returns:
        First is positive samples and then is negative samples
    """
    # feat = np.concatenate([load_positive(dset_name, dset_dir, feat_dir),
    #                        load_negative(dset_name, dset_dir, feat_dir)], axis=0)
    if only_posivite == 0:
        feat_pos = load_positive(dset_name, dset_dir, feat_dir)
        feat_neg = load_negative(dset_name, dset_dir, feat_dir)
        feat = np.concatenate([feat_pos, feat_neg], axis=0)
        true_label = np.concatenate([np.ones(feat_pos.shape[0]), np.zeros(feat_neg.shape[0])], axis=0)
    else:
        feat = load_positive(dset_name, dset_dir, feat_dir)
        true_label = np.ones(feat.shape[0])
    return feat, true_label


# def prepare_ONLY_POS_features(dset_name):
#     pos = pd.read_csv('data/' + dset_name + '/pairs_pos.csv')
#
#     # print(pos)
#     Fvector = read_mat_to_feat('uni_Fvector.mat')
#     pos_Fvector = Fvector.loc[pos.proteinA]
#     # print(Fvector)
#
#     LD = read_mat_to_feat('uni_LD.mat')
#     pos_LD = LD.loc[pos.proteinA]
#     # print(LD)
#
#     APAAC = read_mat_to_feat('LAM_LAI_DSET_HUMAN/uni_APAAC.mat')
#     pos_APAAC = APAAC.loc[pos.proteinA]
#     # print(APAAC)
#
#     pos_feat_out = pd.concat([pos_Fvector, pos_LD, pos_APAAC], axis=1)
#     return pos_feat_out


if __name__ == "__main__":
    X = load_feature_lable('Yeast', feat_dir='../LAM_LAI_DSET_YEAST')
    print(X)
    print(X.shape)
