""" code: thnhan
"""

import os
import pickle

import numpy as np

from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from rotation_forest import RotationForestClassifier
from sklearn.svm import SVC

import matplotlib.pyplot as plt

from hilbert_transform import transform
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

        model = RandomForestClassifier()
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

    fix_len = 80
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
    
    
    # plot_h_p(pssm_NA[0], 50)
    
    print('Data PSSM', pssm_NA.shape, pssm_NB.shape, pssm_PA.shape, pssm_PB.shape)

    print('\nHilbert transforming ...')

    pssm_NA = transform(pssm_NA)
    pssm_NB = transform(pssm_NB)
    pssm_PA = transform(pssm_PA)
    pssm_PB = transform(pssm_PB)

    print('Done')
    
    print('Data Hilbert', pssm_NA.shape, pssm_NB.shape, pssm_PA.shape, pssm_PB.shape)
    
    
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





# Random Forest
# Evaluate model ...
# Prediction
# [[0.09944211 0.90055789]
#  [0.57537261 0.42462739]
#  [0.13884656 0.86115344]
#  ...
#  [0.7109963  0.2890037 ]
#  [0.92912169 0.07087831]
#  [0.91407576 0.08592424]]
# |Accuracy    |Sensitivity |Specificity |Precision   |NPV         |F1-score    |MCC-score   |AUC         |AUPR        |
# |------------|------------|------------|------------|------------|------------|------------|------------|------------|
# |      94.01%|      90.62%|      97.41%|      97.22%|      91.21%|      93.80%|      88.23%|      97.33%|      97.92%|
# Prediction
# [[0.14061091 0.85938909]
#  [0.87403374 0.12596626]
#  [0.46997796 0.53002204]
#  ...
#  [0.92401445 0.07598555]
#  [0.91051393 0.08948607]
#  [0.84007431 0.15992569]]
# |Accuracy    |Sensitivity |Specificity |Precision   |NPV         |F1-score    |MCC-score   |AUC         |AUPR        |
# |------------|------------|------------|------------|------------|------------|------------|------------|------------|
# |      93.61%|      90.71%|      96.51%|      96.30%|      91.22%|      93.42%|      87.37%|      96.98%|      97.66%|
# Prediction
# [[0.00777778 0.99222222]
#  [0.84872034 0.15127966]
#  [0.01333333 0.98666667]
#  ...
#  [0.87056081 0.12943919]
#  [0.92639332 0.07360668]
#  [0.9574127  0.0425873 ]]
# |Accuracy    |Sensitivity |Specificity |Precision   |NPV         |F1-score    |MCC-score   |AUC         |AUPR        |
# |------------|------------|------------|------------|------------|------------|------------|------------|------------|
# |      93.83%|      90.53%|      97.14%|      96.94%|      91.11%|      93.62%|      87.86%|      97.30%|      97.99%|
# Prediction
# [[0.15645503 0.84354497]
#  [0.08829426 0.91170574]
#  [0.0547033  0.9452967 ]
#  ...
#  [0.90554738 0.09445262]
#  [0.74906277 0.25093723]
#  [0.78879533 0.21120467]]
# |Accuracy    |Sensitivity |Specificity |Precision   |NPV         |F1-score    |MCC-score   |AUC         |AUPR        |
# |------------|------------|------------|------------|------------|------------|------------|------------|------------|
# |      94.14%|      90.26%|      98.03%|      97.87%|      90.95%|      93.91%|      88.56%|      96.96%|      97.83%|
# Prediction
# [[0.01474074 0.98525926]
#  [0.77569555 0.22430445]
#  [0.9233955  0.0766045 ]
#  ...
#  [0.87539899 0.12460101]
#  [0.92958971 0.07041029]
#  [0.93349443 0.06650557]]
# |Accuracy    |Sensitivity |Specificity |Precision   |NPV         |F1-score    |MCC-score   |AUC         |AUPR        |
# |------------|------------|------------|------------|------------|------------|------------|------------|------------|
# |      94.46%|      90.43%|      98.48%|      98.35%|      91.15%|      94.22%|      89.20%|      97.48%|      98.11%|

# Final scores (mean)
# |Accuracy       |Sensitivity    |Specificity    |Precision      |NPV            |F1-score       |MCC-score      |AUC            |AUPR           |
# |---------------|---------------|---------------|---------------|---------------|---------------|---------------|---------------|---------------|
# |  94.01%+/-0.29%|  90.51%+/-0.15%|  97.52%+/-0.69%|  97.33%+/-0.71%|  91.13%+/-0.10%|  93.80%+/-0.27%|  88.24%+/-0.62%|  97.21%+/-0.20%|  97.90%+/-0.15%|



# Rotation Forest
# Evaluate model ...
# Prediction
# [[0.  1. ]
#  [0.2 0.8]
#  [0.  1. ]
#  ...
#  [1.  0. ]
#  [1.  0. ]
#  [1.  0. ]]
# |Accuracy    |Sensitivity |Specificity |Precision   |NPV         |F1-score    |MCC-score   |AUC         |AUPR        |
# |------------|------------|------------|------------|------------|------------|------------|------------|------------|
# |      89.10%|      87.04%|      91.15%|      90.77%|      87.55%|      88.87%|      78.26%|      93.98%|      93.52%|
# Prediction
# [[0.    1.   ]
#  [0.225 0.775]
#  [0.6   0.4  ]
#  ...
#  [0.725 0.275]
#  [0.9   0.1  ]
#  [1.    0.   ]]
# |Accuracy    |Sensitivity |Specificity |Precision   |NPV         |F1-score    |MCC-score   |AUC         |AUPR        |
# |------------|------------|------------|------------|------------|------------|------------|------------|------------|
# |      89.05%|      86.33%|      91.78%|      91.30%|      87.03%|      88.75%|      78.22%|      93.42%|      92.72%|
# Prediction
# [[0.   1.  ]
#  [0.95 0.05]
#  [0.   1.  ]
#  ...
#  [1.   0.  ]
#  [0.7  0.3 ]
#  [0.7  0.3 ]]
# |Accuracy    |Sensitivity |Specificity |Precision   |NPV         |F1-score    |MCC-score   |AUC         |AUPR        |
# |------------|------------|------------|------------|------------|------------|------------|------------|------------|
# |      88.56%|      86.24%|      90.88%|      90.44%|      86.85%|      88.29%|      77.21%|      93.18%|      92.75%|
# Prediction
# [[0.         1.        ]
#  [0.         1.        ]
#  [0.         1.        ]
#  ...
#  [0.2        0.8       ]
#  [0.93333333 0.06666667]
#  [0.7        0.3       ]]
# |Accuracy    |Sensitivity |Specificity |Precision   |NPV         |F1-score    |MCC-score   |AUC         |AUPR        |
# |------------|------------|------------|------------|------------|------------|------------|------------|------------|
# |      89.85%|      87.67%|      92.04%|      91.68%|      88.17%|      89.63%|      79.78%|      93.49%|      93.14%|
# Prediction
# [[0.   1.  ]
#  [0.38 0.62]
#  [1.   0.  ]
#  ...
#  [1.   0.  ]
#  [1.   0.  ]
#  [1.   0.  ]]
# |Accuracy    |Sensitivity |Specificity |Precision   |NPV         |F1-score    |MCC-score   |AUC         |AUPR        |
# |------------|------------|------------|------------|------------|------------|------------|------------|------------|
# |      89.18%|      86.23%|      92.14%|      91.63%|      87.00%|      88.85%|      78.50%|      93.24%|      92.62%|

# Final scores (mean)
# |Accuracy       |Sensitivity    |Specificity    |Precision      |NPV            |F1-score       |MCC-score      |AUC            |AUPR           |
# |---------------|---------------|---------------|---------------|---------------|---------------|---------------|---------------|---------------|
# |  89.15%+/-0.41%|  86.70%+/-0.57%|  91.60%+/-0.49%|  91.17%+/-0.49%|  87.32%+/-0.49%|  88.88%+/-0.43%|  78.39%+/-0.82%|  93.46%+/-0.28%|  92.95%+/-0.34%|




# |Accuracy    |Sensitivity |Specificity |Precision   |NPV         |F1-score    |MCC-score   |AUC         |AUPR        |
# |------------|------------|------------|------------|------------|------------|------------|------------|------------|
# |      94.33%|      91.24%|      97.41%|      97.24%|      91.75%|      94.14%|      88.82%|      97.71%|      98.07%|

# |Accuracy    |Sensitivity |Specificity |Precision   |NPV         |F1-score    |MCC-score   |AUC         |AUPR        |
# |------------|------------|------------|------------|------------|------------|------------|------------|------------|
# |      94.15%|      91.33%|      96.96%|      96.78%|      91.79%|      93.98%|      88.43%|      97.36%|      97.84%|

# |Accuracy    |Sensitivity |Specificity |Precision   |NPV         |F1-score    |MCC-score   |AUC         |AUPR        |
# |------------|------------|------------|------------|------------|------------|------------|------------|------------|
# |      93.92%|      90.80%|      97.05%|      96.85%|      91.34%|      93.73%|      88.02%|      97.70%|      98.18%|

# |Accuracy    |Sensitivity |Specificity |Precision   |NPV         |F1-score    |MCC-score   |AUC         |AUPR        |
# |------------|------------|------------|------------|------------|------------|------------|------------|------------|
# |      94.41%|      90.80%|      98.03%|      97.88%|      91.41%|      94.20%|      89.06%|      97.51%|      98.13%|

# |Accuracy    |Sensitivity |Specificity |Precision   |NPV         |F1-score    |MCC-score   |AUC         |AUPR        |
# |------------|------------|------------|------------|------------|------------|------------|------------|------------|
# |      94.64%|      90.79%|      98.48%|      98.35%|      91.45%|      94.42%|      89.54%|      97.94%|      98.41%|

# Final scores (mean)
# |Accuracy       |Sensitivity    |Specificity    |Precision      |NPV            |F1-score       |MCC-score      |AUC            |AUPR           |
# |---------------|---------------|---------------|---------------|---------------|---------------|---------------|---------------|---------------|
# |  94.29%+/-0.24%|  90.99%+/-0.24%|  97.59%+/-0.58%|  97.42%+/-0.61%|  91.55%+/-0.19%|  94.09%+/-0.23%|  88.77%+/-0.52%|  97.64%+/-0.20%|  98.12%+/-0.18%|