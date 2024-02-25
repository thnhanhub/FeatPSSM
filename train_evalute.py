""" code: thnhan
"""

import os
import pickle

import numpy as np

from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from rotation_forest import RotationForestClassifier

import matplotlib.pyplot as plt

from hilbert_transform import transform
from utils.plot_utils.plot_roc_pr_curve import plot_folds
from utils.report_result import my_cv_report, print_metrics

from scipy.signal import hilbert, hilbert2

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

        model = RandomForestClassifier(n_estimators=100, min_samples_leaf=1, min_samples_split=10)
        # model = RotationForestClassifier()
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

    fix_len = 50
    np.random.seed(42)  # 42

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

    print('\nHilbert transforming ...')

    pssm_NA = transform(pssm_NA)
    pssm_NB = transform(pssm_NB)
    pssm_PA = transform(pssm_PA)
    pssm_PB = transform(pssm_PB)

    print('Done')

    print("\nMake X, y ...")

    pssm_NA = np.reshape(pssm_NA, (-1, pssm_NA.shape[1] * pssm_NA.shape[2]))
    pssm_NB = np.reshape(pssm_NB, (-1, pssm_NB.shape[1] * pssm_NB.shape[2]))
    pssm_PA = np.reshape(pssm_PA, (-1, pssm_PA.shape[1] * pssm_PA.shape[2]))
    pssm_PB = np.reshape(pssm_PB, (-1, pssm_PB.shape[1] * pssm_PB.shape[2]))

    pos = np.concatenate([pssm_PA, pssm_PB], axis=1)
    neg = np.concatenate([pssm_NA, pssm_NB], axis=1)
    
    print("Data interactions", pos.shape, neg.shape)
    print("\n")

    X = np.concatenate([pos, neg], axis=0)
    y = np.array([1] * len(pos) + [0] * len(neg))

    print("X, y:", X.shape, y.shape)

    print("\nEvaluate model ...")
    eval_model(X, y)

# Final scores (mean)
# |Accuracy       |Sensitivity    |Specificity    |Precision      |NPV            |F1-score       |MCC-score      |AUC            |AUPR           |
# |---------------|---------------|---------------|---------------|---------------|---------------|---------------|---------------|---------------|
# |  94.00%+/-0.35%|  90.56%+/-0.32%|  97.44%+/-0.59%|  97.26%+/-0.62%|  91.17%+/-0.28%|  93.79%+/-0.36%|  88.22%+/-0.73%|  96.94%+/-0.26%|  97.70%+/-0.20%|

# Final scores (mean)
# |Accuracy       |Sensitivity    |Specificity    |Precision      |NPV            |F1-score       |MCC-score      |AUC            |AUPR           |
# |---------------|---------------|---------------|---------------|---------------|---------------|---------------|---------------|---------------|
# |  94.05%+/-0.29%|  90.65%+/-0.12%|  97.44%+/-0.66%|  97.26%+/-0.68%|  91.25%+/-0.08%|  93.84%+/-0.28%|  88.30%+/-0.62%|  97.02%+/-0.20%|  97.75%+/-0.16%|

# Final scores (mean)
# |Accuracy       |Sensitivity    |Specificity    |Precision      |NPV            |F1-score       |MCC-score      |AUC            |AUPR           |
# |---------------|---------------|---------------|---------------|---------------|---------------|---------------|---------------|---------------|
# |  93.97%+/-0.39%|  90.56%+/-0.25%|  97.37%+/-0.57%|  97.18%+/-0.60%|  91.16%+/-0.26%|  93.75%+/-0.40%|  88.14%+/-0.81%|  96.97%+/-0.15%|  97.72%+/-0.16%|


# Final scores (mean)
# |Accuracy       |Sensitivity    |Specificity    |Precision      |NPV            |F1-score       |MCC-score      |AUC            |AUPR           |
# |---------------|---------------|---------------|---------------|---------------|---------------|---------------|---------------|---------------|
# |  93.92%+/-0.37%|  90.24%+/-0.12%|  97.60%+/-0.73%|  97.42%+/-0.77%|  90.91%+/-0.12%|  93.69%+/-0.36%|  88.09%+/-0.79%|  97.01%+/-0.23%|  97.72%+/-0.20%|


# Final scores (mean)
# |Accuracy       |Sensitivity    |Specificity    |Precision      |NPV            |F1-score       |MCC-score      |AUC            |AUPR           |
# |---------------|---------------|---------------|---------------|---------------|---------------|---------------|---------------|---------------|
# |  93.95%+/-0.34%|  90.40%+/-0.25%|  97.50%+/-0.51%|  97.31%+/-0.53%|  91.04%+/-0.24%|  93.73%+/-0.35%|  88.12%+/-0.70%|  96.96%+/-0.19%|  97.70%+/-0.19%|