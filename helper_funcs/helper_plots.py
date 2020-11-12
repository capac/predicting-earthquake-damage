#!/usr/bin/env python

import matplotlib.pyplot as plt
from pathlib import PurePath
# from numpy import fill_diagonal


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
    # fill_diagonal(norm_conf_mx, 0)
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    g = ax.matshow(norm_conf_mx, cmap=plt.cm.coolwarm)
    # Loop over data dimensions and create text annotations.
    for i in range(norm_conf_mx.shape[1]):
        for j in range(norm_conf_mx.shape[0]):
            ax.text(j, i, f'{norm_conf_mx[i, j]:.3f}', ha='center', va='center', color='w', fontsize=12)
    ax.set_xlabel('Predicted damage level', fontsize=14)
    ax.set_ylabel('Actual damage level', fontsize=14)
    ax.set_xticks(range(0, 3))
    ax.set_xticklabels(['Low', 'Medium', 'High'], fontsize=12)
    ax.set_yticks(range(0, 3))
    ax.set_yticklabels(['Low', 'Medium', 'High'], fontsize=12)
    cbar = plt.colorbar(g, fraction=0.041, pad=0.04)
    cbar.ax.set_yticklabels([f'{v:.1f}' for v in cbar.ax.get_yticks().tolist()], fontsize=10)
    # cbar.set_label('Prediction gradient', fontsize=16)
    fig.tight_layout()
    plt.savefig(PurePath(model_dir) / 'confusion-matrix-heatmap.png', dpi=288)
