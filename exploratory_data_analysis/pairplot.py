#!/usr/bin/env python

import pandas as pd
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

num_attrib_list = ['count_floors_pre_eq', 'age', 'area_percentage', 'height_percentage',
                   'count_families', 'damage_grade']

num_attrib_df = train_values_df.join(train_labels_df)[num_attrib_list]

label_font_size = 14
g = sns.PairGrid(num_attrib_df, hue='damage_grade', palette='tab10', height=3.0, aspect=1.2)
g = g.map_diag(plt.hist)
g = g.map_offdiag(sns.regplot, scatter=False)
g = g.map_offdiag(plt.scatter, s=60, alpha=0.6)
g = g.add_legend()
g._legend.set_title('Damage Level', prop={'size': label_font_size})
for txt, lb in zip(g._legend.texts, ['Low', 'Medium', 'High']):
    txt.set_text(lb)
    txt.set_fontsize(label_font_size)

xlabels, ylabels = [], []

for ax in g.axes[-1, :]:
    xlabel = ax.xaxis.get_label_text()
    xlabels.append(xlabel)
for ax in g.axes[:, 0]:
    ylabel = ax.yaxis.get_label_text()
    ylabels.append(ylabel)
for i in range(len(xlabels)):
    for j in range(len(ylabels)):
        g.axes[j, i].xaxis.set_label_text(xlabels[i])
        g.axes[j, i].xaxis.label.set_size(label_font_size)
        g.axes[j, i].tick_params(axis='x', which='major', labelsize=label_font_size)
        g.axes[j, i].yaxis.set_label_text(ylabels[j])
        g.axes[j, i].yaxis.label.set_size(label_font_size)
        g.axes[j, i].tick_params(axis='y', which='major', labelsize=label_font_size)

plt.tight_layout(rect=(0, 0, 0.92, 1))
plt.savefig(PurePath.joinpath(plot_dir, 'pairplot-with-reg.png'), dpi=288)
# plt.show()
