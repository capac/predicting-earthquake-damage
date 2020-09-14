#!/use/bin/env python

from pathlib import PurePath
from joblib import dump, load
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


def run_clf(X_train, X_test, y_train, y_test, clf_list, model_dir):
    t0 = time()
    list_files = [x for x in model_dir.iterdir() if x.is_file]
    accuracy_score_dict = dict()
    for item in clf_list:
        model_file = PurePath.joinpath(model_dir, item.name+'.sav')
        if model_file in list_files:
            t1 = time()
            model_clf = load(model_file)
            accuracy_score_dict[model_clf.__class__.__name__] = print_accuracy(model_clf, t1, X_test, y_test.values)
        else:
            t2 = time()
            item.classifier.fit(X_train, y_train.values)
            dump(item.classifier, model_file)
            accuracy_score_dict[item.classifier.__class__.__name__] = print_accuracy(item.classifier, t2, X_test, y_test.values)
    print(f'\nTotal time elasped: {time() - t0:.4f} sec')