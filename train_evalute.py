"""
Performance on Yeast core datasets:
1. Using 5-fold cross-validation on "Yeast core" datasets.
2. Params selection

@author: thnhan
"""
import os
import pickle

import numpy as np
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model
from gensim.models import Word2Vec
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from models.net import net
from utils.plot_utils.plot_roc_pr_curve import plot_folds
from utils.report_result import print_metrics, my_cv_report
import matplotlib.pyplot as plt


def fix_len_pssm(lst_PSSM, fixlen, fixvalue=0.0):
    new = []
    for ii in range(len(lst_PSSM)):
        if len(lst_PSSM[ii]) >= fixlen:
            new.append(lst_PSSM[ii][:fixlen])
        else:
            temp = np.full(shape=(fixlen - len(lst_PSSM[ii]), 20), fill_value=fixvalue)
            temp = np.concatenate([lst_PSSM[ii], temp], axis=0)
            new.append(temp)
    return np.array(new)


def eval_model(pairs, labels):
    skf = StratifiedKFold(n_splits=5, random_state=48, shuffle=True)
    scores = []
    hists = []
    cv_prob_Y, cv_test_y = [], []

    for i, (tr_inds, te_inds) in enumerate(skf.split(pairs, labels)):
        # Fix do dai
        feat_A = np.concatenate([fix_len_pssm(pssm_PA, seq_lens[i]),
                                 fix_len_pssm(pssm_NA, seq_lens[i])], axis=0)
        feat_B = np.concatenate([fix_len_pssm(pssm_PB, seq_lens[i]),
                                 fix_len_pssm(pssm_NB, seq_lens[i])], axis=0)
        feat = np.concatenate([feat_A, feat_B], axis=1)

        print(feat_A.shape)
        print(feat.shape)

        print("\nFold", i)
        Y = to_categorical(labels)

        tr_X_A, tr_X_B = feat_A[tr_inds], feat_B[tr_inds]
        te_X_A, te_X_B = feat_A[te_inds], feat_B[te_inds]
        tr_Y, te_Y = Y[tr_inds], Y[te_inds]

        print(tr_X_A.shape)
        print(te_X_A.shape)

        # scal = StandardScaler().fit(tr_X)
        # tr_X = scal.transform(tr_X)
        # te_X = scal.transform(te_X)


        # ====== DEF MODEL
        if os.path.exists('OurModel_trained_on_Yeastcore_fold' + str(i) + '.h5'):
            model = load_model('OurModel_trained_on_Yeastcore_fold' + str(i) + '.h5')
        else:
            model = net(seq_lens[i])
            opt = Adam(decay=0.001)
            model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

            # ====== FIT MODEL
            hist = model.fit([tr_X_A, tr_X_B], tr_Y,
                             batch_size=512,  # 64
                             epochs=2,  # 45
                             verbose=1)
            hists.append(hist)

            # ====== SAVE MODEL
            model.save("OurModel_trained_on_Yeastcore_fold" + str(i) + ".h5")

        # ====== REPORT
        prob_Y = model.predict([te_X_A, te_X_B])
        # te_y = np.argmax(te_Y, axis=1)
        scr = print_metrics(np.argmax(te_Y, axis=1), prob_Y)
        scores.append(scr)

        cv_prob_Y.append(prob_Y[:, 1])
        cv_test_y.append(np.argmax(te_Y, axis=1))

    # ====== FINAL REPORT
    print("\nFinal scores (mean)")
    scores_array = np.array(scores)
    my_cv_report(scores_array)

    plot_folds(plt, cv_test_y, cv_prob_Y)
    plt.show()
    return hists


if __name__ == "__main__":
    pssm_NA = pickle.load(open("yestcore_neg_A_pssm.pkl", "rb"))
    pssm_NB = pickle.load(open("yestcore_neg_B_pssm.pkl", "rb"))
    pssm_PA = pickle.load(open("yestcore_pos_A_pssm.pkl", "rb"))
    pssm_PB = pickle.load(open("yestcore_pos_B_pssm.pkl", "rb"))

    pairs = np.arange(len(pssm_PA) + len(pssm_NA))
    labels = np.array([1] * len(pssm_PA) + [0] * len(pssm_NA))
    print(pairs)

    seq_lens = [554, 561, 559, 561, 560]

    # print(model2.summary())
    eval_model(pairs, labels)
