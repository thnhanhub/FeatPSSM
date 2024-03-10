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

import matplotlib.pyplot as plt

from hilbert_transform import transform, extracted_features
from utils.plot_utils.plot_roc_pr_curve import plot_folds
from utils.report_result import my_cv_report, print_metrics

from scipy.signal import hilbert, hilbert2
from plot_hilbert_pssm import plot_h_p

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

        # model = RandomForestClassifier()
        model = ExtraTreesClassifier()
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

    print("Fix length:", fix_len)

    print("\nLoad PSSM data ...")
    
    pssm_NA = pickle.load(open("neg_A_pssm.pkl", "rb"))
    pssm_NB = pickle.load(open("neg_B_pssm.pkl", "rb"))
    pssm_PA = pickle.load(open("pos_A_pssm.pkl", "rb"))
    pssm_PB = pickle.load(open("pos_B_pssm.pkl", "rb"))

    pssm_NA = fix_len_pssm(pssm_NA, fix_len)
    pssm_NB = fix_len_pssm(pssm_NB, fix_len)
    pssm_PA = fix_len_pssm(pssm_PA, fix_len)
    pssm_PB = fix_len_pssm(pssm_PB, fix_len)
    
    sum_pssm_NA = np.sum(pssm_NA, axis=1)
    sum_pssm_NB = np.sum(pssm_NA, axis=1)
    sum_pssm_PA = np.sum(pssm_NA, axis=1)
    sum_pssm_PB = np.sum(pssm_NA, axis=1)
    # plot_h_p(pssm_NA[0], 50)
    
    print('Data PSSM', pssm_NA.shape, pssm_NB.shape, pssm_PA.shape, pssm_PB.shape)

    print('\nHilbert transforming ...')

    features_NA, pssm_NA = transform(pssm_NA)
    features_NB, pssm_NB = transform(pssm_NB)
    features_PA, pssm_PA = transform(pssm_PA)
    features_PB, pssm_PB = transform(pssm_PB)
    
    sum_hilb_NA = np.sum(pssm_NA, axis=1)
    sum_hilb_NB = np.sum(pssm_NA, axis=1)
    sum_hilb_PA = np.sum(pssm_NA, axis=1)
    sum_hilb_PB = np.sum(pssm_NA, axis=1)

    print('Done')
    
    print('Data Hilbert', pssm_NA.shape, pssm_NB.shape, pssm_PA.shape, pssm_PB.shape)
    
    print("\nMake X, y ...")

    pssm_NA = np.reshape(pssm_NA, (-1, pssm_NA.shape[1] * pssm_NA.shape[2]))
    pssm_NB = np.reshape(pssm_NB, (-1, pssm_NB.shape[1] * pssm_NB.shape[2]))
    pssm_PA = np.reshape(pssm_PA, (-1, pssm_PA.shape[1] * pssm_PA.shape[2]))
    pssm_PB = np.reshape(pssm_PB, (-1, pssm_PB.shape[1] * pssm_PB.shape[2]))
    
    features_NA = np.reshape(features_NA, (-1, features_NA.shape[1] * features_NA.shape[2]))
    features_NB = np.reshape(features_NB, (-1, features_NB.shape[1] * features_NB.shape[2]))
    features_PA = np.reshape(features_PA, (-1, features_PA.shape[1] * features_PA.shape[2]))
    features_PB = np.reshape(features_PB, (-1, features_PB.shape[1] * features_PB.shape[2]))
    
    pssm_NA = np.concatenate([pssm_NA, sum_hilb_NA, sum_pssm_NA, features_NA], axis=1)
    pssm_NB = np.concatenate([pssm_NB, sum_hilb_NB, sum_pssm_NB, features_NB], axis=1)
    pssm_PA = np.concatenate([pssm_PA, sum_hilb_PA, sum_pssm_PA, features_PA], axis=1)
    pssm_PB = np.concatenate([pssm_PB, sum_hilb_PB, sum_pssm_PB, features_PB], axis=1)

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


# Random Forest không biến đổi Hilbert
# |Accuracy    |Sensitivity |Specificity |Precision   |NPV         |F1-score    |MCC-score   |AUC         |AUPR        |
# |------------|------------|------------|------------|------------|------------|------------|------------|------------|
# |      94.01%|      90.62%|      97.41%|      97.22%|      91.21%|      93.80%|      88.23%|      97.33%|      97.92%|

# |Accuracy    |Sensitivity |Specificity |Precision   |NPV         |F1-score    |MCC-score   |AUC         |AUPR        |
# |------------|------------|------------|------------|------------|------------|------------|------------|------------|
# |      93.61%|      90.71%|      96.51%|      96.30%|      91.22%|      93.42%|      87.37%|      96.98%|      97.66%|

# |Accuracy    |Sensitivity |Specificity |Precision   |NPV         |F1-score    |MCC-score   |AUC         |AUPR        |
# |------------|------------|------------|------------|------------|------------|------------|------------|------------|
# |      93.83%|      90.53%|      97.14%|      96.94%|      91.11%|      93.62%|      87.86%|      97.30%|      97.99%|

# |Accuracy    |Sensitivity |Specificity |Precision   |NPV         |F1-score    |MCC-score   |AUC         |AUPR        |
# |------------|------------|------------|------------|------------|------------|------------|------------|------------|
# |      94.14%|      90.26%|      98.03%|      97.87%|      90.95%|      93.91%|      88.56%|      96.96%|      97.83%|

# |Accuracy    |Sensitivity |Specificity |Precision   |NPV         |F1-score    |MCC-score   |AUC         |AUPR        |
# |------------|------------|------------|------------|------------|------------|------------|------------|------------|
# |      94.46%|      90.43%|      98.48%|      98.35%|      91.15%|      94.22%|      89.20%|      97.48%|      98.11%|

# Final scores (mean)
# |Accuracy       |Sensitivity    |Specificity    |Precision      |NPV            |F1-score       |MCC-score      |AUC            |AUPR           |
# |---------------|---------------|---------------|---------------|---------------|---------------|---------------|---------------|---------------|
# |  94.01%+/-0.29%|  90.51%+/-0.15%|  97.52%+/-0.69%|  97.33%+/-0.71%|  91.13%+/-0.10%|  93.80%+/-0.27%|  88.24%+/-0.62%|  97.21%+/-0.20%|  97.90%+/-0.15%|


# Random Forest sử dụng Hilbert
# |Accuracy    |Sensitivity |Specificity |Precision   |NPV         |F1-score    |MCC-score   |AUC         |AUPR        |
# |------------|------------|------------|------------|------------|------------|------------|------------|------------|
# |      97.63%|      96.16%|      99.11%|      99.08%|      96.27%|      97.60%|      95.31%|      99.87%|      99.86%|

# |Accuracy    |Sensitivity |Specificity |Precision   |NPV         |F1-score    |MCC-score   |AUC         |AUPR        |
# |------------|------------|------------|------------|------------|------------|------------|------------|------------|
# |      97.14%|      95.35%|      98.93%|      98.89%|      95.51%|      97.09%|      94.34%|      99.81%|      99.81%|

# |Accuracy    |Sensitivity |Specificity |Precision   |NPV         |F1-score    |MCC-score   |AUC         |AUPR        |
# |------------|------------|------------|------------|------------|------------|------------|------------|------------|
# |      97.18%|      94.73%|      99.64%|      99.62%|      94.97%|      97.11%|      94.48%|      99.87%|      99.87%|

# |Accuracy    |Sensitivity |Specificity |Precision   |NPV         |F1-score    |MCC-score   |AUC         |AUPR        |
# |------------|------------|------------|------------|------------|------------|------------|------------|------------|
# |      96.74%|      93.66%|      99.82%|      99.81%|      94.02%|      96.63%|      93.65%|      99.88%|      99.89%|

# |Accuracy    |Sensitivity |Specificity |Precision   |NPV         |F1-score    |MCC-score   |AUC         |AUPR        |
# |------------|------------|------------|------------|------------|------------|------------|------------|------------|
# |      97.68%|      95.53%|      99.82%|      99.81%|      95.72%|      97.62%|      95.44%|      99.94%|      99.94%|

# Final scores (mean)
# |Accuracy       |Sensitivity    |Specificity    |Precision      |NPV            |F1-score       |MCC-score      |AUC            |AUPR           |
# |---------------|---------------|---------------|---------------|---------------|---------------|---------------|---------------|---------------|
# |  97.27%+/-0.35%|  95.08%+/-0.85%|  99.46%+/-0.37%|  99.44%+/-0.39%|  95.30%+/-0.76%|  97.21%+/-0.37%|  94.64%+/-0.66%|  99.88%+/-0.04%|  99.87%+/-0.04%|


# Rotation Forest không biến đổi Hilbert
# |Accuracy    |Sensitivity |Specificity |Precision   |NPV         |F1-score    |MCC-score   |AUC         |AUPR        |
# |------------|------------|------------|------------|------------|------------|------------|------------|------------|
# |      89.10%|      87.04%|      91.15%|      90.77%|      87.55%|      88.87%|      78.26%|      93.98%|      93.52%|

# |Accuracy    |Sensitivity |Specificity |Precision   |NPV         |F1-score    |MCC-score   |AUC         |AUPR        |
# |------------|------------|------------|------------|------------|------------|------------|------------|------------|
# |      89.05%|      86.33%|      91.78%|      91.30%|      87.03%|      88.75%|      78.22%|      93.42%|      92.72%|

# |Accuracy    |Sensitivity |Specificity |Precision   |NPV         |F1-score    |MCC-score   |AUC         |AUPR        |
# |------------|------------|------------|------------|------------|------------|------------|------------|------------|
# |      88.56%|      86.24%|      90.88%|      90.44%|      86.85%|      88.29%|      77.21%|      93.18%|      92.75%|

# |Accuracy    |Sensitivity |Specificity |Precision   |NPV         |F1-score    |MCC-score   |AUC         |AUPR        |
# |------------|------------|------------|------------|------------|------------|------------|------------|------------|
# |      89.85%|      87.67%|      92.04%|      91.68%|      88.17%|      89.63%|      79.78%|      93.49%|      93.14%|

# |Accuracy    |Sensitivity |Specificity |Precision   |NPV         |F1-score    |MCC-score   |AUC         |AUPR        |
# |------------|------------|------------|------------|------------|------------|------------|------------|------------|
# |      89.18%|      86.23%|      92.14%|      91.63%|      87.00%|      88.85%|      78.50%|      93.24%|      92.62%|

# Final scores (mean)
# |Accuracy       |Sensitivity    |Specificity    |Precision      |NPV            |F1-score       |MCC-score      |AUC            |AUPR           |
# |---------------|---------------|---------------|---------------|---------------|---------------|---------------|---------------|---------------|
# |  89.15%+/-0.41%|  86.70%+/-0.57%|  91.60%+/-0.49%|  91.17%+/-0.49%|  87.32%+/-0.49%|  88.88%+/-0.43%|  78.39%+/-0.82%|  93.46%+/-0.28%|  92.95%+/-0.34%|

# Rotation Forest sử dụng biến đổi Hilbert
# |Accuracy    |Sensitivity |Specificity |Precision   |NPV         |F1-score    |MCC-score   |AUC         |AUPR        |
# |------------|------------|------------|------------|------------|------------|------------|------------|------------|
# |      94.55%|      91.51%|      97.59%|      97.43%|      92.00%|      94.38%|      89.26%|      96.53%|      96.15%|

# |Accuracy    |Sensitivity |Specificity |Precision   |NPV         |F1-score    |MCC-score   |AUC         |AUPR        |
# |------------|------------|------------|------------|------------|------------|------------|------------|------------|
# |      95.17%|      92.14%|      98.21%|      98.10%|      92.59%|      95.02%|      90.52%|      97.06%|      96.59%|

# |Accuracy    |Sensitivity |Specificity |Precision   |NPV         |F1-score    |MCC-score   |AUC         |AUPR        |
# |------------|------------|------------|------------|------------|------------|------------|------------|------------|
# |      94.73%|      91.24%|      98.21%|      98.08%|      91.81%|      94.54%|      89.67%|      97.78%|      97.58%|

# |Accuracy    |Sensitivity |Specificity |Precision   |NPV         |F1-score    |MCC-score   |AUC         |AUPR        |
# |------------|------------|------------|------------|------------|------------|------------|------------|------------|
# |      95.71%|      92.67%|      98.75%|      98.67%|      93.09%|      95.58%|      91.59%|      97.77%|      97.71%|

# |Accuracy    |Sensitivity |Specificity |Precision   |NPV         |F1-score    |MCC-score   |AUC         |AUPR        |
# |------------|------------|------------|------------|------------|------------|------------|------------|------------|
# |      95.44%|      92.75%|      98.12%|      98.02%|      93.13%|      95.31%|      91.01%|      96.99%|      96.40%|

# Final scores (mean)
# |Accuracy       |Sensitivity    |Specificity    |Precision      |NPV            |F1-score       |MCC-score      |AUC            |AUPR           |
# |---------------|---------------|---------------|---------------|---------------|---------------|---------------|---------------|---------------|
# |  95.12%+/-0.43%|  92.06%+/-0.61%|  98.18%+/-0.37%|  98.06%+/-0.39%|  92.52%+/-0.54%|  94.97%+/-0.45%|  90.41%+/-0.85%|  97.23%+/-0.48%|  96.89%+/-0.63%|

# Extra Tree sử dụng Hilbert
# |Accuracy    |Sensitivity |Specificity |Precision   |NPV         |F1-score    |MCC-score   |AUC         |AUPR        |
# |------------|------------|------------|------------|------------|------------|------------|------------|------------|
# |      97.54%|      95.62%|      99.46%|      99.44%|      95.78%|      97.49%|      95.16%|      99.91%|      99.90%|

# |Accuracy    |Sensitivity |Specificity |Precision   |NPV         |F1-score    |MCC-score   |AUC         |AUPR        |
# |------------|------------|------------|------------|------------|------------|------------|------------|------------|
# |      96.87%|      94.64%|      99.11%|      99.06%|      94.87%|      96.80%|      93.84%|      99.85%|      99.85%|

# |Accuracy    |Sensitivity |Specificity |Precision   |NPV         |F1-score    |MCC-score   |AUC         |AUPR        |
# |------------|------------|------------|------------|------------|------------|------------|------------|------------|
# |      97.23%|      94.73%|      99.73%|      99.72%|      94.98%|      97.16%|      94.58%|      99.92%|      99.91%|

# |Accuracy    |Sensitivity |Specificity |Precision   |NPV         |F1-score    |MCC-score   |AUC         |AUPR        |
# |------------|------------|------------|------------|------------|------------|------------|------------|------------|
# |      96.78%|      93.74%|      99.82%|      99.81%|      94.10%|      96.68%|      93.74%|      99.94%|      99.93%|

# |Accuracy    |Sensitivity |Specificity |Precision   |NPV         |F1-score    |MCC-score   |AUC         |AUPR        |
# |------------|------------|------------|------------|------------|------------|------------|------------|------------|
# |      97.41%|      94.99%|      99.82%|      99.81%|      95.23%|      97.34%|      94.93%|      99.96%|      99.96%|

# Final scores (mean)
# |Accuracy       |Sensitivity    |Specificity    |Precision      |NPV            |F1-score       |MCC-score      |AUC            |AUPR           |
# |---------------|---------------|---------------|---------------|---------------|---------------|---------------|---------------|---------------|
# |  97.17%+/-0.30%|  94.74%+/-0.61%|  99.59%+/-0.27%|  99.57%+/-0.29%|  94.99%+/-0.55%|  97.10%+/-0.31%|  94.45%+/-0.57%|  99.91%+/-0.03%|  99.91%+/-0.03%|