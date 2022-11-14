
import sys
import pandas as pd
import numpy as np
#import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn import metrics
from sklearn.model_selection import train_test_split, KFold, cross_validate, GridSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

import pickle

#from rpy2.robjects.packages import importr
#from rpy2.robjects import FloatVector

#utils = importr('utils')
#utils.install_packages('PRROC')

#prroc = importr('PRROC')


def do_RF_paramtuning(X_train, y_train, class_wt):
    reslist = []
    aucmetric = metrics.make_scorer(get_aucpr_R, needs_proba=True)
    metric_idx=1  # index where AUC is stored
    for cval in [10, 50, 100, 200, 500, 1000]:
        if(class_wt>0):
            logreg = RandomForestClassifier(n_estimators=cval, class_weight="balanced")
        else:
            logreg = RandomForestClassifier(n_estimators=cval)
        cv_results = cross_validate(logreg, X_train, y_train, cv=5, scoring=aucmetric)
        reslist.append((cval, np.mean(cv_results['test_score'])))

    print(*reslist, sep='\n')
    reslist = np.asarray(reslist)
    bestid = np.where(reslist[:,metric_idx]==max(reslist[:,metric_idx]))[0][0]

    if(class_wt>0):
        clf = RandomForestClassifier(n_estimators=int(reslist[bestid,0]), class_weight="balanced")
    else:
        clf = RandomForestClassifier(n_estimators=int(reslist[bestid,0]))
    clf = clf.fit(X_train, y_train)
    return clf


def do_logreg_paramtuning(X_train, y_train, class_wt):
    reslist = []
    metric_idx=1  # index where AUC is stored
    aucmetric = metrics.make_scorer(get_aucpr_R, needs_proba=True)
    #for cval in [1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1, 10, 100, 1000, 10**5]:
    for cval in [1e-5, 1e-3, 0.1, 1, 10, 100]:
        print('Doing',cval)
        logreg = LogisticRegression(random_state=0, penalty='l2', C=cval, max_iter=20000, solver='lbfgs')   # class_weight={0:(1-class_wt+0.1), 1:1}
        cv_results = cross_validate(logreg, X_train, y_train, cv=5, scoring=aucmetric)
        reslist.append((cval, np.mean(cv_results['test_score'])))
    print(*reslist, sep='\n')
    reslist = np.asarray(reslist)
    bestid = np.where(reslist[:,metric_idx]==max(reslist[:,metric_idx]))[0][0]
    clf = LogisticRegression(random_state=0, penalty='l2', C=reslist[bestid,0], max_iter=20000, solver='lbfgs')
    clf = clf.fit(X_train, y_train)
    return clf

def normalize_train_test(X_train, X_test, X_cov):
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    X_cov = scaler.transform(X_cov)
    return X_train, X_test, X_cov

def impute_train_test(X_train, X_test):
    #replace -8888 values with Nan and then use simple imputer 
    imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
    imp_mean.fit(X_train)
    X_train = imp_mean.transform(X_train)
    X_test = imp_mean.transform(X_test)
    return X_train, X_test

def imputeX(X):
    #replace -8888 values with Nan and then use simple imputer 
    imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
    imp_mean.fit(X)
    X = imp_mean.transform(X)
    return X

def get_aucpr(y_true, y_pred, pos_label=1):
    #fpr, tpr, thresholds = sklearn.metrics.roc_curve(y_true, pred, pos_label=2)
    precision, recall, thresholds = metrics.precision_recall_curve(y_true, y_pred, pos_label)
    auc_val = metrics.auc(recall, precision)
    return auc_val

def get_pr_curve(y_true, y_pred):
    precision, recall, thresholds = metrics.precision_recall_curve(y_true, y_pred)
    return recall, precision

def get_auc(labels, preds, pos_label=1):
    fpr, tpr, _ = metrics.roc_curve(labels, preds, pos_label)
    return metrics.auc(fpr, tpr)

def binarize(y_pred):
    return [int(x >= 0.5) for x in y_pred]


def get_fmax(y_true, y_pred, pos_label=1):
    #fpr, tpr, thresholds = sklearn.metrics.roc_curve(y_true, pred, pos_label=2)
    precision, recall, thresholds = metrics.precision_recall_curve(y_true, y_pred, pos_label)
    #f = plt.figure()
    #plt.plot(precision, recall)
    #plt.show()
    #f.savefig("pr_plot.pdf", bbox_inches='tight')
    return compute_fmax(precision, recall)

def get_aucpr_R(y_true, y_pred, pos_label=1):
    return get_aucpr(y_true, y_pred, pos_label)
    #prCurve = prroc.pr_curve(scores_class0 = FloatVector(list(y_pred)),
    #          weights_class0 = FloatVector(list(y_true)), curve=True)
    #return prCurve[2][0]

def get_auc_R(y_true, y_pred, pos_label=1):
    return get_auc(y_true, y_pred, pos_label)
    #rocCurve = prroc.roc_curve(scores_class0 = FloatVector(list(y_pred)),
    #          weights_class0 = FloatVector(list(y_true)), curve=True)
    #return rocCurve[1][0]

def compute_eval_measures(scores, positives, negatives=None,
        track_pos=False, track_neg=False):
    """
    Compute the precision and false-positive rate at each change in recall (true-positive rate)
    *scores*: array containing a score for each node
    *positives*: indices of positive nodes
    *negatives*: if negatives are given, then the FP will only be from the set of negatives given
    *track_pos*: if specified, track the score and rank of the positive nodes,
        and return a tuple of the node ids in order of their score, their score, their idx, and 1/-1 for pos/neg
    *track_neg*: also track the score and rank of the negative nodes
    """
    #f1_score = metrics.f1score(positives, 
    #num_unknowns = len(scores) - len(positives) 
    positives = set(positives)
    check_negatives = False
    if negatives is not None:
        check_negatives = True
        negatives = set(negatives)
    else:
        print("TODO. Treating all non-positives as negatives not yet implemented.")
    # compute the precision and recall at each change in recall
    # use np.argsort
    nodes_sorted_by_scores = np.argsort(scores)[::-1]
    precision = []
    recall = []
    fpr = []
    pos_neg_stats = []  # tuple containing the node, score, idx, pos/neg assign., and the # positives assigned so far
    # TP is the # of correctly predicted positives so far
    TP = 0
    FP = 0
    rec = 0
    for i, n in enumerate(nodes_sorted_by_scores):
        # TODO this could be slow if there are many positives
        if n in positives:
            TP += 1
            # precisions is the # of true positives / # true positives + # of false positives (or the total # of predictions)
            precision.append(TP / float(TP + FP))
            # recall is the # of recovered positives TP / TP + FN (total # of positives)
            rec = TP / float(len(positives))
            recall.append(rec)
            # fpr is the FP / FP + TN
            fpr.append((rec, FP / float(len(negatives))))
            if track_pos:
                pos_neg_stats.append((n, scores[n], i, 1, TP+FP))
        elif check_negatives is False or n in negatives:
            FP += 1
            # store the prec and rec even though recall doesn't change since the AUPRC will be affected
            precision.append(TP / float(TP + FP))
            recall.append(TP / float(len(positives)))
            fpr.append((rec, FP / float(len(negatives))))
            if track_neg:
                pos_neg_stats.append((n, scores[n], i, -1, TP+FP))
        #else:
        #    continue

    # TODO shouldn't happen
    if len(precision) == 0:
        precision.append(0)
        recall.append(1)

    #print(precision[0], recall[0], fpr[0])

    if track_pos or track_neg:
        return precision, recall, fpr, pos_neg_stats
    else:
        return precision, recall, fpr


def compute_early_prec(prec, rec, recall_vals=[0.1, 0.2, 0.5], pred_ranks=None, num_pos=None):
    early_prec_values = []
    for curr_recall in recall_vals:
        # if a k recall is specified, get the precision at the recall which is k * # ann in the left-out species
        if True:
            curr_recall = float(curr_recall)
            # find the first precision value where the recall is >= the specified recall
            for p, r in zip(prec, rec):
                if r <= curr_recall:
                    early_prec_values.append(prev_p)
                    break
                prev_p = p
    #print(early_prec_values)
    return early_prec_values


def get_early_prec(y_true, y_pred, pos_label=1):
    #fpr, tpr, thresholds = sklearn.metrics.roc_curve(y_true, pred, pos_label=2)
    precision, recall, thresholds = metrics.precision_recall_curve(y_true, y_pred, pos_label)
    return compute_early_prec(precision, recall)


def compute_fmax(prec, rec, fmax_idx=False):
    """
    *fmax_idx*: also return the index at which the harmonic mean is the greatest
    """
    f_measures = []
    for i in range(len(prec)):
        p, r = prec[i], rec[i]
        if p+r == 0:
            harmonic_mean = 0
        else:
            # see https://en.wikipedia.org/wiki/Harmonic_mean#Two_numbers
            harmonic_mean = (2*p*r)/(p+r)
        f_measures.append(harmonic_mean)
    if fmax_idx:
        idx = np.argmax(np.asarray(f_measures))
        return max(f_measures), idx
    else:
        return max(f_measures)


def save_model(ebm, model_file):
    model_pkl = open(model_file, 'wb')
    pickle.dump(ebm,model_pkl)
    model_pkl.close()
