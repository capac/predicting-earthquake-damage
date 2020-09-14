#!/usr/bin/env python

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path, PurePath
import os

home = os.environ['HOME']
project_root_dir = Path(home) / 'Programming/Python/driven-data/predicting-earthquake-damage'
plot_dir = project_root_dir / 'exploratory_data_analysis/plots'
data_dir = project_root_dir / 'data'

train_values_file = data_dir / 'train_values.csv'
train_labels_file = data_dir / 'train_labels.csv'
train_values_df = pd.read_csv(train_values_file, index_col='building_id')
train_labels_df = pd.read_csv(train_labels_file, index_col='building_id')

list_tuple_categories = [('foundation_type', 'position'), ('legal_ownership_status', 'land_surface_condition'), ('roof_type', 'ground_floor_type'), ('other_floor_type', 'plan_configuration')]

def bar_plot(ax, df, col):
    for i in range(0, 2):
        df_col = df[col[i]].value_counts(ascending=False)
        df_col = df_col/df_col.sum()*100
        df_col = df_col.reset_index().rename(columns={'index': 'features'})
        ax[i].bar(df_col['features'], df_col[col[i]], color=colors, edgecolor='k')        
        if all(isinstance(tick, np.float64) for tick in ax[i].get_xticks()):
            ax[i].set_xticks(range(0, 2))
            ax[i].set_xticklabels(['No', 'Yes'])
        plt.setp(ax[i].get_xticklabels(), ha="right", rotation_mode="anchor", rotation=0, fontsize=14)
        plt.setp(ax[i].get_yticklabels(), fontsize=14)
        ax[i].set_ylabel('Percent (%)', fontsize=14)
        ax[i].set_title(col[i], fontsize=16)


fig, axes = plt.subplots(nrows=4, ncols=2, figsize=(14, 12))
colors = plt.cm.tab10.colors
for ax, col in zip(axes, list_tuple_categories):
    bar_plot(ax, train_values_df, col)
plt.tight_layout()
plt.savefig(PurePath.joinpath(plot_dir, 'categorical-bar-plot-1.png'), dpi=288)

list_tuple_binary_cat_attrib = [('has_superstructure_adobe_mud', 'has_superstructure_mud_mortar_stone'), ('has_superstructure_stone_flag', 'has_superstructure_cement_mortar_stone'), ('has_superstructure_mud_mortar_brick', 'has_superstructure_cement_mortar_brick'), ('has_superstructure_timber', 'has_superstructure_bamboo'), ('has_superstructure_rc_non_engineered', 'has_superstructure_rc_engineered'), ('has_superstructure_other', 'has_secondary_use'), ('has_secondary_use_agriculture', 'has_secondary_use_hotel'), ('has_secondary_use_rental', 'has_secondary_use_institution'), ('has_secondary_use_school', 'has_secondary_use_industry'), ('has_secondary_use_health_post', 'has_secondary_use_gov_office'), ('has_secondary_use_use_police', 'has_secondary_use_other')]

fig, axes = plt.subplots(nrows=11, ncols=2, figsize=(14, 30))
for ax, col in zip(axes, list_tuple_binary_cat_attrib):
    bar_plot(ax, train_values_df, col)
plt.tight_layout()
plt.savefig(PurePath.joinpath(plot_dir, 'categorical-bar-plot-2.png'), dpi=288)
