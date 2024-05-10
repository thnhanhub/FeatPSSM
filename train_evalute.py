""" code: thnhan
"""

import os
import pickle

import numpy as np

from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from rotation_forest import RotationForestClassifier
from sklearn.svm import SVC
from deepforest import CascadeForestClassifier

import matplotlib.pyplot as plt

from hilbert_transform import transform, extracted_features
from utils.plot_utils.plot_roc_pr_curve import plot_folds
from utils.report_result import my_cv_report, print_metrics

from scipy.signal import hilbert, hilbert2
from plot_hilbert_pssm import plot_h_p
from scipy.ndimage import rotate

def fix_len_pssm(lst_PSSM, fixlen, fixvalue=0.0):
    new = []
    for ii in range(len(lst_PSSM)):
        if len(lst_PSSM[ii]) >= fixlen:
            new.append(lst_PSSM[ii][:fixlen])
        else:
            temp = np.full(shape=(fixlen - len(lst_PSSM[ii]), 20), fill_value=fixvalue)
            temp = np.concatenate([lst_PSSM[ii], temp], axis=0)
    return np.array(new)

def eval_model(X_, y_):
    skf = StratifiedKFold(n_splits=5, random_state=48, shuffle=True)
    scores = []

    cv_hists = []
    cv_prob_Y, cv_test_y = [], []

    for i, (tr_inds, te_inds) in enumerate(skf.split(X_, y_)):
        train_X, train_y = X_[tr_inds], y_[tr_inds]
        test_X, test_y = X_[te_inds], y_[te_inds]
        
        model = CascadeForestClassifier(n_estimators=2)
        estimators = [RandomForestClassifier(), RotationForestClassifier()]
        model.set_estimator(estimators)
        
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
    # fix_len = 50
    
    np.random.seed(44)  # 42


    print("\nLoad PSSM data ...")
    
    pssm_NA = pickle.load(open("neg_A_pssm.pkl", "rb"))
    pssm_NB = pickle.load(open("neg_B_pssm.pkl", "rb"))
    pssm_PA = pickle.load(open("pos_A_pssm.pkl", "rb"))
    pssm_PB = pickle.load(open("pos_B_pssm.pkl", "rb"))
    
    # pssm_NA = fix_len_pssm(pssm_NA, fix_len)
    # pssm_NB = fix_len_pssm(pssm_NB, fix_len)
    # pssm_PA = fix_len_pssm(pssm_PA, fix_len)
    # pssm_PB = fix_len_pssm(pssm_PB, fix_len)
    
    # plot_h_p(pssm_NA[0], 20)

    print("\nExtracting features... ")
    
    pssm_NA = transform(pssm_NA)
    pssm_NB = transform(pssm_NB)
    pssm_PA = transform(pssm_PA)
    pssm_PB = transform(pssm_PB)
    
    # pssm_NA = np.reshape(pssm_NA, (-1, pssm_NA.shape[1] * pssm_NA.shape[2]))
    # pssm_NB = np.reshape(pssm_NB, (-1, pssm_NB.shape[1] * pssm_NB.shape[2]))
    # pssm_PA = np.reshape(pssm_PA, (-1, pssm_PA.shape[1] * pssm_PA.shape[2]))
    # pssm_PB = np.reshape(pssm_PB, (-1, pssm_PB.shape[1] * pssm_PB.shape[2]))
    
    print('Data ', pssm_NA.shape, pssm_NB.shape, pssm_PA.shape, pssm_PB.shape)

    pos = np.concatenate([pssm_PA, pssm_PB], axis=1)
    neg = np.concatenate([pssm_NA, pssm_NB], axis=1)
    
    print("Data interactions ", pos.shape, neg.shape)
    print("\n")

    X = np.concatenate([pos, neg], axis=0)
    y = np.array([1] * len(pos) + [0] * len(neg))

    print("X, y:", X.shape, y.shape)

    print("\nEvaluate model ...")
    eval_model(X, y)


# 1
# Final scores (mean)
# |Accuracy       |Sensitivity    |Specificity    |Precision      |NPV            |F1-score       |MCC-score      |AUC            |AUPR           |
# |---------------|---------------|---------------|---------------|---------------|---------------|---------------|---------------|---------------|
# |  95.24%+/-0.38%|  92.92%+/-0.51%|  97.55%+/-0.46%|  97.43%+/-0.47%|  93.24%+/-0.47%|  95.12%+/-0.39%|  90.57%+/-0.76%|  98.49%+/-0.19%|  98.82%+/-0.12%|