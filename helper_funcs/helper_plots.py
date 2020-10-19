#!/usr/bin/env python

import matplotlib.pyplot as plt
from pathlib import PurePath
from numpy import fill_diagonal


def xgb_clf_eval(model, model_dir):
    results = model.evals_result()
    epochs = len(results['validation_0']['merror'])
    x_axis = range(0, epochs)
    # plot log loss
    fig, ax = plt.subplots(2, 1, figsize=(10, 8))
    ax[0].plot(x_axis, results['validation_0']['mlogloss'], label='Train')
    ax[0].plot(x_axis, results['validation_1']['mlogloss'], label='Test')
    ax[0].legend()
    ax[0].set_ylabel('Log Loss')
    ax[0].set_title('XGBoost Log Loss')
    # plot classification error
    ax[1].plot(x_axis, results['validation_0']['merror'], label='Train')
    ax[1].plot(x_axis, results['validation_1']['merror'], label='Test')
    ax[1].legend()
    ax[1].set_ylabel('Classification Error')
    ax[1].set_title('XGBoost Classification Error')
    fig.tight_layout()
    plt.savefig(PurePath(model_dir) / 'logloss_clferr.png', dpi=144)


def conf_mx_heat_plot(conf_mx, model_dir):
    row_sums = conf_mx.sum(axis=1, keepdims=True)
    norm_conf_mx = conf_mx/row_sums
    fill_diagonal(norm_conf_mx, 0)
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    ax.matshow(norm_conf_mx, cmap=plt.cm.coolwarm)
    ax.set_xlabel('Predicted', fontsize=14)
    ax.set_ylabel('Actual', fontsize=14)
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(['low', 'medium', 'high'], fontsize=12)
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(['low', 'medium', 'high'], fontsize=12)
    fig.tight_layout()
    plt.savefig(PurePath(model_dir) / 'conf_mx_heat_plot.png', dpi=144)
