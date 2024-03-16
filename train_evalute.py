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
    return np.array(new)

def get_hilb(X_):
    X_hat_ = []
    for x in X_:
        X_hat_.append(hilbert(x, axis=1).imag)
    return np.array(X_hat_)

def get_features(lst_PSSM):
    features = []

    hilb_feat, hilb_X =  transform(lst_PSSM)
    for ii in range(len(lst_PSSM)):
        features.append(np.sum(lst_PSSM[ii], axis=0))
    features = np.concatenate([features, hilb_X, hilb_feat], axis=1)
    return np.array(features)

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


    print("\nLoad PSSM data ...")
    
    pssm_NA = pickle.load(open("neg_A_pssm.pkl", "rb"))
    pssm_NB = pickle.load(open("neg_B_pssm.pkl", "rb"))
    pssm_PA = pickle.load(open("pos_A_pssm.pkl", "rb"))
    pssm_PB = pickle.load(open("pos_B_pssm.pkl", "rb"))
    
    # plot_h_p(pssm_NA[0], 50)
    # print('Data PSSM ', pssm_NA.shape, pssm_NB.shape, pssm_PA.shape, pssm_PB.shape)
    
    print("\nExtracting features... ")
    
    features_NA = get_features(pssm_NA)
    features_NB = get_features(pssm_NB)
    features_PA = get_features(pssm_PA)
    features_PB = get_features(pssm_PB)
    
    print('Data features ', features_NA.shape, features_NB.shape, features_PA.shape, features_PB.shape)

    print("Fix length:", fix_len)

    pssm_NA = fix_len_pssm(pssm_NA, fix_len)
    pssm_NB = fix_len_pssm(pssm_NB, fix_len)
    pssm_PA = fix_len_pssm(pssm_PA, fix_len)
    pssm_PB = fix_len_pssm(pssm_PB, fix_len)
    
    print("\nGet Hilbert data... ")
    
    pssm_NA = get_hilb(pssm_NA)
    pssm_NB = get_hilb(pssm_NB)
    pssm_PA = get_hilb(pssm_PA)
    pssm_PB = get_hilb(pssm_PB)
    
    pssm_NA = np.reshape(pssm_NA, (-1, pssm_NA.shape[1] * pssm_NA.shape[2]))
    pssm_NB = np.reshape(pssm_NB, (-1, pssm_NB.shape[1] * pssm_NB.shape[2]))
    pssm_PA = np.reshape(pssm_PA, (-1, pssm_PA.shape[1] * pssm_PA.shape[2]))
    pssm_PB = np.reshape(pssm_PB, (-1, pssm_PB.shape[1] * pssm_PB.shape[2]))
    
    print('Data hilbert ', pssm_NA.shape, pssm_NB.shape, pssm_PA.shape, pssm_PB.shape)
    
    # print('Data features ', pssm_NA.shape, pssm_NB.shape, pssm_PA.shape, pssm_PB.shape)
        
    pssm_NA = np.concatenate([pssm_NA, features_NA], axis=1)
    pssm_NB = np.concatenate([pssm_NB, features_NB], axis=1)
    pssm_PA = np.concatenate([pssm_PA, features_PA], axis=1)
    pssm_PB = np.concatenate([pssm_PB, features_PB], axis=1)
    

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
# |      94.33%|      91.15%|      97.50%|      97.33%|      91.68%|      94.14%|      88.83%|      98.01%|      98.33%|

# |Accuracy    |Sensitivity |Specificity |Precision   |NPV         |F1-score    |MCC-score   |AUC         |AUPR        |
# |------------|------------|------------|------------|------------|------------|------------|------------|------------|
# |      94.46%|      91.96%|      96.96%|      96.80%|      92.34%|      94.32%|      89.03%|      97.95%|      98.33%|

# |Accuracy    |Sensitivity |Specificity |Precision   |NPV         |F1-score    |MCC-score   |AUC         |AUPR        |
# |------------|------------|------------|------------|------------|------------|------------|------------|------------|
# |      94.59%|      91.24%|      97.94%|      97.80%|      91.79%|      94.41%|      89.39%|      98.14%|      98.50%|

# |Accuracy    |Sensitivity |Specificity |Precision   |NPV         |F1-score    |MCC-score   |AUC         |AUPR        |
# |------------|------------|------------|------------|------------|------------|------------|------------|------------|
# |      95.08%|      91.51%|      98.66%|      98.56%|      92.07%|      94.90%|      90.40%|      97.72%|      98.28%|

# |Accuracy    |Sensitivity |Specificity |Precision   |NPV         |F1-score    |MCC-score   |AUC         |AUPR        |
# |------------|------------|------------|------------|------------|------------|------------|------------|------------|
# |      95.08%|      91.59%|      98.57%|      98.46%|      92.15%|      94.90%|      90.39%|      98.33%|      98.63%|

# Final scores (mean)
# |Accuracy       |Sensitivity    |Specificity    |Precision      |NPV            |F1-score       |MCC-score      |AUC            |AUPR           |
# |---------------|---------------|---------------|---------------|---------------|---------------|---------------|---------------|---------------|
# |  94.71%+/-0.32%|  91.49%+/-0.28%|  97.93%+/-0.64%|  97.79%+/-0.67%|  92.01%+/-0.24%|  94.53%+/-0.31%|  89.61%+/-0.67%|  98.03%+/-0.20%|  98.41%+/-0.13%|


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
# |      89.59%|      87.85%|      91.33%|      91.02%|      88.26%|      89.40%|      79.23%|      92.44%|      90.68%|

# |Accuracy    |Sensitivity |Specificity |Precision   |NPV         |F1-score    |MCC-score   |AUC         |AUPR        |
# |------------|------------|------------|------------|------------|------------|------------|------------|------------|
# |      89.72%|      88.83%|      90.62%|      90.45%|      89.03%|      89.63%|      79.46%|      91.80%|      89.66%|

# |Accuracy    |Sensitivity |Specificity |Precision   |NPV         |F1-score    |MCC-score   |AUC         |AUPR        |
# |------------|------------|------------|------------|------------|------------|------------|------------|------------|
# |      89.23%|      85.97%|      92.49%|      91.97%|      86.83%|      88.87%|      78.63%|      92.57%|      90.82%|

# |Accuracy    |Sensitivity |Specificity |Precision   |NPV         |F1-score    |MCC-score   |AUC         |AUPR        |
# |------------|------------|------------|------------|------------|------------|------------|------------|------------|
# |      90.30%|      88.92%|      91.68%|      91.45%|      89.21%|      90.17%|      80.63%|      93.09%|      91.73%|

# |Accuracy    |Sensitivity |Specificity |Precision   |NPV         |F1-score    |MCC-score   |AUC         |AUPR        |
# |------------|------------|------------|------------|------------|------------|------------|------------|------------|
# |      89.63%|      87.66%|      91.60%|      91.25%|      88.13%|      89.42%|      79.32%|      92.79%|      91.25%|

# Final scores (mean)
# |Accuracy       |Sensitivity    |Specificity    |Precision      |NPV            |F1-score       |MCC-score      |AUC            |AUPR           |
# |---------------|---------------|---------------|---------------|---------------|---------------|---------------|---------------|---------------|
# |  89.69%+/-0.35%|  87.84%+/-1.07%|  91.54%+/-0.60%|  91.23%+/-0.50%|  88.29%+/-0.84%|  89.50%+/-0.42%|  79.45%+/-0.65%|  92.54%+/-0.43%|  90.83%+/-0.69%|


# Extra Tree sử dụng Hilbert
# |Accuracy    |Sensitivity |Specificity |Precision   |NPV         |F1-score    |MCC-score   |AUC         |AUPR        |
# |------------|------------|------------|------------|------------|------------|------------|------------|------------|
# |      94.82%|      91.42%|      98.21%|      98.08%|      91.97%|      94.63%|      89.84%|      97.73%|      98.11%|

# |Accuracy    |Sensitivity |Specificity |Precision   |NPV         |F1-score    |MCC-score   |AUC         |AUPR        |
# |------------|------------|------------|------------|------------|------------|------------|------------|------------|
# |      94.59%|      91.60%|      97.59%|      97.43%|      92.07%|      94.43%|      89.35%|      97.62%|      98.22%|

# |Accuracy    |Sensitivity |Specificity |Precision   |NPV         |F1-score    |MCC-score   |AUC         |AUPR        |
# |------------|------------|------------|------------|------------|------------|------------|------------|------------|
# |      94.73%|      91.33%|      98.12%|      97.99%|      91.88%|      94.54%|      89.66%|      97.90%|      98.19%|

# |Accuracy    |Sensitivity |Specificity |Precision   |NPV         |F1-score    |MCC-score   |AUC         |AUPR        |
# |------------|------------|------------|------------|------------|------------|------------|------------|------------|
# |      94.95%|      91.24%|      98.66%|      98.55%|      91.84%|      94.76%|      90.15%|      97.59%|      98.27%|

# |Accuracy    |Sensitivity |Specificity |Precision   |NPV         |F1-score    |MCC-score   |AUC         |AUPR        |
# |------------|------------|------------|------------|------------|------------|------------|------------|------------|
# |      95.22%|      91.59%|      98.84%|      98.75%|      92.17%|      95.03%|      90.67%|      98.13%|      98.56%|

# Final scores (mean)
# |Accuracy       |Sensitivity    |Specificity    |Precision      |NPV            |F1-score       |MCC-score      |AUC            |AUPR           |
# |---------------|---------------|---------------|---------------|---------------|---------------|---------------|---------------|---------------|
# |  94.86%+/-0.21%|  91.44%+/-0.14%|  98.28%+/-0.44%|  98.16%+/-0.46%|  91.99%+/-0.12%|  94.68%+/-0.21%|  89.93%+/-0.45%|  97.80%+/-0.20%|  98.27%+/-0.16%|