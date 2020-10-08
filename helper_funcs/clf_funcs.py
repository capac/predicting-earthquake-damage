#!/use/bin/env python

from pathlib import PurePath
from joblib import dump, load
from sklearn.metrics import accuracy_score, classification_report, f1_score
from collections import namedtuple
from helper_funcs.xgb_plots import xgb_clf_eval
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
            # plots only if XGBoostClassifier is trained with early_stopping_rounds option
            if model_clf.__class__.__name__ == 'XGBClassifier':
                xgb_clf_eval(model_clf, model_dir)
        else:
            t2 = time()
            if item.classifier.__class__.__name__ == 'XGBClassifier':
                item.classifier.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_val, y_val)],
                                    early_stopping_rounds=20, eval_metric=['merror', 'mlogloss'],
                                    verbose=True)
                xgb_clf_eval(item.classifier, model_dir)
            else:
                item.classifier.fit(X_train, y_train)
            dump(item.classifier, model_file, compress=3)
            accuracy_score_dict[item.classifier.__class__.__name__] =\
                print_accuracy(item.classifier, X_val, y_val, start_time=t2, print_output=True)
    print(f'\nTotal time elasped: {time() - t0:.4f} sec')
    return accuracy_score_dict


def grid_search_func(X_train, y_train, grid_search, joblib_file, model_dir=None, X_val=None, y_val=None):
    t0 = time()
    if joblib_file.is_file():
        grid_search = load(joblib_file)
        if grid_search.best_estimator_.__class__.__name__ == 'XGBClassifier':
            xgb_clf_eval(grid_search.best_estimator_, model_dir)
    else:
        if grid_search.__class__.__name__ == 'RandomizedSearchCV':
            grid_search.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_val, y_val)],
                            early_stopping_rounds=20, eval_metric=['merror', 'mlogloss'],
                            verbose=True)
            xgb_clf_eval(grid_search.best_estimator_, model_dir)
        else:
            grid_search.fit(X_train, y_train)
        dump(grid_search, joblib_file, compress=3)
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


def run_ensemble_clf(X_train, X_val, y_train, y_val, voting_clf, model_dir):
    t0 = time()
    joblib_file = PurePath.joinpath(model_dir, 'voting_clf.sav')
    if joblib_file.is_file():
        voting_clf = load(joblib_file)
    else:
        voting_clf.fit(X_train, y_train)
        dump(voting_clf, joblib_file)
    y_pred = voting_clf.predict(X_val)
    print(f'''Micro-averaged F1 score for VotingClassifier: {f1_score(y_val, y_pred, average='micro'):.8f}''')
    print(f'\nTime elapsed: {time() - t0:.4f} sec')
