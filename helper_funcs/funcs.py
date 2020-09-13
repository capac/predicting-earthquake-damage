#!/use/bin/env python

from sklearn.metrics import accuracy_score, classification_report, f1_score
from collections import namedtuple
from time import time
import sys


def clf_func(clf_dict):
    clf_list = []
    for item in clf_dict.items():
        clf_ntuple = namedtuple('clf_ntuple', ['name', 'classifier'])
        clf_list.append(clf_ntuple(name=item[0], classifier=item[1]))
    return clf_list


def print_accuracy(model_clf, t, X_test, y_test):
    y_pred = model_clf.predict(X_test)
    acc_score = accuracy_score(y_pred, y_test)
    sys.stdout.write(f'\nAccuracy score for {model_clf.__class__.__name__}: {acc_score:.6f}')
    sys.stdout.write(f'\nTime elapsed: {time() - t:.4f} sec')
    sys.stdout.write(f'\nClassification report:\n{classification_report(y_pred, y_test, digits=4)}')
    return acc_score
