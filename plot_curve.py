import matplotlib.pyplot as plt
import numpy
from sklearn.datasets import make_blobs
from sklearn.metrics import precision_recall_curve, auc, average_precision_score
from sklearn.model_selection import KFold
from sklearn.svm import SVC

FOLDS = 5

X, y = make_blobs(n_samples=2000, n_features=2, centers=2, cluster_std=10.0,
                  random_state=12345)


def draw_cv_pr_curve(y_true_, y_prob_, figsize=(10, 5), show_plt=True):
    f, axes = plt.subplots(1, 2, figsize=figsize)

    axes[0].scatter(X[y == 0, 0], X[y == 0, 1], color='blue', s=2, label='y=0')
    axes[0].scatter(X[y != 0, 0], X[y != 0, 1], color='red', s=2, label='y=1')
    axes[0].set_xlabel('X[:,0]')
    axes[0].set_ylabel('X[:,1]')
    axes[0].set_title('Positive and Negartive samples')
    axes[0].legend(loc='lower left', fontsize='small')

    n_folds = len(y_true_)

    y_real = []
    y_proba = []
    for ii in range(n_folds):
        y_true_ii, y_prob_ii = y_true_[ii], y_prob_[ii][:, 1]

        precision, recall, _ = precision_recall_curve(y_true_ii, y_prob_ii)
        lab = 'Fold %d AUC=%.4f' % (ii + 1,
                                    auc(recall, precision),
                                    # average_precision_score(ytest, pred_proba[:, 1])
                                    )
        axes[1].step(recall, precision, label=lab)
        y_real.append(y_true_ii)
        y_proba.append(y_prob_ii)

    y_real = numpy.concatenate(y_real)
    y_proba = numpy.concatenate(y_proba)
    precision, recall, _ = precision_recall_curve(y_real, y_proba)
    lab = 'Overall AUC=%.4f' % (
        auc(recall, precision)
        # average_precision_score(y_real, y_proba)
    )
    axes[1].step(recall, precision, label=lab, lw=2, color='black')
    axes[1].set_xlabel('Recall')
    axes[1].set_ylabel('Precision')
    axes[1].set_title('Precision-Recall Curve')
    axes[1].legend(loc='lower left', fontsize='small')

    f.tight_layout()
    f.savefig('result_ave_precision.svg')
    if show_plt:
        plt.show()
    return plt


k_fold = KFold(n_splits=FOLDS, shuffle=True, random_state=12345)
predictor = SVC(kernel='linear', C=1.0, probability=True, random_state=12345)

k_fold_y_true = []
k_fold_y_prob = []
for i, (train_index, test_index) in enumerate(k_fold.split(X)):
    Xtrain, Xtest = X[train_index], X[test_index]
    ytrain, ytest = y[train_index], y[test_index]
    k_fold_y_true.append(ytest)

    predictor.fit(Xtrain, ytrain)
    k_fold_y_prob.append(predictor.predict_proba(Xtest))

plt = draw_cv_pr_curve(k_fold_y_true, k_fold_y_prob)
