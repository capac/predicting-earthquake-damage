#!/use/bin/env python

from pathlib import PurePath
from joblib import dump, load
from sklearn.metrics import accuracy_score, classification_report, f1_score
from collections import namedtuple
from time import time
import sys
from numpy import sqrt


def clf_func(clf_dict):
    clf_list = []
    for item in clf_dict.items():
        clf_ntuple = namedtuple('clf_ntuple', ['name', 'classifier'])
        clf_list.append(clf_ntuple(name=item[0], classifier=item[1]))
    return clf_list


def print_accuracy(model_clf, X_val, y_val, start_time=None, print_output=False):
    # NOTICE: returns valid time differences only if start_time != None
    y_pred = model_clf.predict(X_val)
    acc_score = accuracy_score(y_pred, y_val)
    micro_averaged_f1_score = f1_score(y_val, y_pred, average='micro')
    #  only prints if print_output=True
    if print_output:
        sys.stdout.write(f'\nAccuracy score for {model_clf.__class__.__name__}: {acc_score:.8f}')
        sys.stdout.write(f'\nMicro-averaged F1 score for {model_clf.__class__.__name__}: {micro_averaged_f1_score:.8f}')
        sys.stdout.write(f'\nTime elapsed: {time() - start_time:.4f} sec')
        sys.stdout.write(f'\nClassification report:\n{classification_report(y_pred, y_val, digits=4)}')
    return acc_score


def run_clf(X_train, X_val, y_train, y_val, clf_list, model_dir):
    t0 = time()
    list_files = [x for x in model_dir.iterdir() if x.is_file]
    accuracy_score_dict = dict()
    for item in clf_list:
        model_file = PurePath.joinpath(model_dir, item.name+'.sav')
        if model_file in list_files:
            t1 = time()
            model_clf = load(model_file)
            accuracy_score_dict[model_clf.__class__.__name__] =\
                print_accuracy(model_clf, X_val, y_val, start_time=t1, print_output=True)
        else:
            t2 = time()
            item.classifier.fit(X_train, y_train)
            dump(item.classifier, model_file)
            accuracy_score_dict[item.classifier.__class__.__name__] =\
                print_accuracy(item.classifier, X_val, y_val, start_time=t2, print_output=True)
    print(f'\nTotal time elasped: {time() - t0:.4f} sec')
    return accuracy_score_dict


def grid_search_func(X_train, y_train, grid_search, joblib_file):
    t0 = time()
    if joblib_file.is_file():
        grid_search = load(joblib_file)
    else:
        grid_search.fit(X_train, y_train)
        dump(grid_search, joblib_file)
    print(f'Best parameters for grid search: {grid_search.best_params_}\n')
    print(f'Best estimator for grid search: {grid_search.best_estimator_}\n')
    print(f'Time elapsed: {time() - t0:.4f} sec')
    return grid_search


def grid_results(grid_search, num):
    cvres = grid_search.cv_results_
    best_fit_models = [(sqrt(mean_score), params) for mean_score,
                       params in zip(cvres['mean_test_score'], cvres['params'])]
    best_fit_models.sort(key=lambda x: x[0], reverse=True)
    print('List of best-fit models sorted by micro F1 score:')
    for rmse, params in best_fit_models[:num]:
        print(f'{rmse} {params}')
