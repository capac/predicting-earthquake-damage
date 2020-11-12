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
train_values_df = pd.read_csv(train_values_file, index_col='building_id')

num_attrib_list = ['count_floors_pre_eq', 'age', 'area_percentage', 'height_percentage', 'count_families']

fig, axes = plt.subplots(figsize=(10, 6))
mask = np.zeros_like(train_values_df[num_attrib_list].corr())
mask[np.tril_indices_from(mask)] = True
hm = sns.heatmap(data=train_values_df[num_attrib_list].corr(), cmap='Spectral', ax=axes,
                 annot=True, fmt='1.4f', mask=mask, annot_kws={'size': 14})
axes.tick_params(labelsize=14)
cbar = hm.collections[0].colorbar
cbar.ax.tick_params(labelsize=14)
plt.setp(axes.get_xticklabels(), ha='right', rotation_mode='anchor', rotation=45)
plt.setp(axes.get_yticklabels(), ha='right', rotation_mode='anchor')
plt.tight_layout()
plt.grid(True, linestyle='--')
plt.savefig(PurePath.joinpath(plot_dir, 'correlation.png'), dpi=288)
