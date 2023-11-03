""" code: thnhan
"""

import os
import pickle

import numpy as np

from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

import matplotlib.pyplot as plt

from utils.plot_utils.plot_roc_pr_curve import plot_folds
from utils.report_result import my_cv_report, print_metrics


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


def eval_model(X_, y_):
    skf = StratifiedKFold(n_splits=5, random_state=48, shuffle=True)
    scores = []

    cv_hists = []
    cv_prob_Y, cv_test_y = [], []

    for i, (tr_inds, te_inds) in enumerate(skf.split(X_, y_)):
        train_X, train_y = X_[tr_inds], y_[tr_inds]
        test_X, test_y = X_[te_inds], y_[te_inds]

        model = RandomForestClassifier(500)

        train_hist = model.fit(train_X, train_y)

        """ Report ... """
        prob_y = model.predict_proba(test_X)
        print("Prediction")
        print(prob_y)
        scr = print_metrics(test_y, prob_y)
        scores.append(scr)

        cv_prob_Y.append(prob_y)
        cv_test_y.append(test_y)

    # ====== FINAL REPORT
    print("\nFinal scores (mean)")
    scores_array = np.array(scores)
    my_cv_report(scores_array)

    plot_folds(plt, cv_test_y, cv_prob_Y)
    plt.show()
    return cv_hists


if __name__ == "__main__":
    print("\nParameters ...")

    fix_len = 600
    print("Fix legnth:", fix_len)

    print("\nLoad PSSM data ...")

    pssm_NA = pickle.load(open("neg_A_pssm.pkl", "rb"))
    pssm_NB = pickle.load(open("neg_B_pssm.pkl", "rb"))
    pssm_PA = pickle.load(open("pos_A_pssm.pkl", "rb"))
    pssm_PB = pickle.load(open("pos_B_pssm.pkl", "rb"))

    pssm_NA = fix_len_pssm(pssm_NA, fix_len)
    pssm_NB = fix_len_pssm(pssm_NB, fix_len)
    pssm_PA = fix_len_pssm(pssm_PA, fix_len)
    pssm_PB = fix_len_pssm(pssm_PB, fix_len)

    print('Data PSSM', pssm_NA.shape, pssm_NB.shape, pssm_PA.shape, pssm_PB.shape)

    print("\nMake X, y ...")

    pssm_NA = np.reshape(pssm_NA, (-1, fix_len * 20))
    pssm_NB = np.reshape(pssm_NB, (-1, fix_len * 20))
    pssm_PA = np.reshape(pssm_PA, (-1, fix_len * 20))
    pssm_PB = np.reshape(pssm_PB, (-1, fix_len * 20))

    pos = np.concatenate([pssm_PA, pssm_PB], axis=1)
    neg = np.concatenate([pssm_NA, pssm_NB], axis=1)
    X = np.concatenate([pos, neg], axis=0)

    y = np.array([1] * len(pos) + [0] * len(neg))
    print("X, y:", X.shape, y.shape)

    print("\nEvaluate model ...")
    eval_model(X, y)
