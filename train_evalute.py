""" code: thnhan
"""

import os
import pickle

import numpy as np
from sklearn.model_selection import StratifiedKFold
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
        < TODO >

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
    pssm_NA = pickle.load(open("neg_A_pssm.pkl", "rb"))
    pssm_NB = pickle.load(open("neg_B_pssm.pkl", "rb"))
    pssm_PA = pickle.load(open("pos_A_pssm.pkl", "rb"))
    pssm_PB = pickle.load(open("pos_B_pssm.pkl", "rb"))

    pairs = np.arange(len(pssm_PA) + len(pssm_NA))
    labels = np.array([1] * len(pssm_PA) + [0] * len(pssm_NA))
    print(pairs)

    seq_lens = 600
    eval_model(pairs, labels)
