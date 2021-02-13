#!/usr/bin/env python

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import ticker
from pathlib import Path
import os

home = os.environ['HOME']
project_root_dir = Path(home) / 'Programming/Python/machine-learning-exercises/driven-data/predicting-earthquake-damage'
plot_dir = project_root_dir / 'exploratory_data_analysis/plots'
data_dir = project_root_dir / 'data'
train_labels_file = data_dir / 'train_labels.csv'
train_labels_df = pd.read_csv(train_labels_file, index_col='building_id')

fig, axes = plt.subplots(figsize=(10, 6))
df = train_labels_df['damage_grade'].value_counts(ascending=False).sort_index()
axes.bar(df.index, df.values, color=plt.cm.tab10.colors, edgecolor='k')
axes.set_xlabel('Level', fontsize=14)
axes.set_ylabel('Number of damaged buildings (in units of $10^4$)', fontsize=14)
axes.set_title('Building damage by level', fontsize=16)
axes.set_xticks(df.index)
axes.set_xticklabels(['Low', 'Medium', 'High'], fontsize=14)
plt.setp(axes.get_yticklabels(), fontsize=14)
ticks = ticker.FuncFormatter(lambda x, _: '{0:g}'.format(x/1e4))
axes.yaxis.set_major_formatter(ticks)
fig.savefig(plot_dir / 'damage-level-by-grade.png', dpi=288, bbox_inches='tight')
