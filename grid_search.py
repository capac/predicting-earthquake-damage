#!/usr/bin/env python

import os
from pandas import read_csv
from pathlib import Path, PurePath
from helper_funcs.funcs import grid_search_func, grid_results, print_accuracy
from helper_funcs.data_preparation import prepare_data, feature_pipeline
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score
from joblib import load
from numpy import prod
# from sklearn.ensemble import VotingClassifier

home = os.environ['HOME']
project_root_dir = Path(home) / 'Programming/Python/driven-data/predicting-earthquake-damage'
plot_dir = project_root_dir / 'exploratory_data_analysis/plots'
data_dir = project_root_dir / 'data'
train_labels_file = data_dir / 'train_labels.csv'
train_labels_df = read_csv(train_labels_file, index_col='building_id')

data_dir = Path('./data')
model_dir = Path('./models')

# data frame creation
data_frame_list = []
data_file_list = ['train_values.csv', 'test_values.csv', 'train_labels.csv']
for data_file in data_file_list:
    data_frame_list.append(read_csv(data_dir / data_file, index_col='building_id'))
train_values_df, test_values_df, train_labels_df = data_frame_list

# convert object and numerical data types in train_values_df, test_values_df to category data types
train_values_df, train_labels_df, test_values_df, num_attrib, \
    cat_attrib = prepare_data(train_values_df, test_values_df, train_labels_df)

# pipeline to place median for NaNs and normalize data
prepared_train_values_df = feature_pipeline(train_values_df, num_attrib, cat_attrib)

# generating stratified training and validation data sets
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
for train_index, val_index in sss.split(prepared_train_values_df, train_labels_df):
    X_strat_train = prepared_train_values_df[train_index]
    y_strat_train = train_labels_df.iloc[train_index]
    X_strat_val = prepared_train_values_df[val_index]
    y_strat_val = train_labels_df.iloc[val_index]
y_strat_train, y_strat_val = y_strat_train.iloc[:, 0], y_strat_val.iloc[:, 0]  # type: ignore

# generating training and test data sets
# X_train, X_test, y_train, y_test = train_test_split(prepared_train_values_df, train_labels_df,
#                                                     test_size=0.3, random_state=42)
# y_train, y_test = y_train.iloc[:, 0], y_test.iloc[:, 0]

# grid search setup on XGBClassifier
xgb_clf = XGBClassifier(n_jobs=-1, verbosity=1, tree_method='hist')
xgb_params = {'max_depth': [10, 15, 20], 'n_estimators': [100, 200]}
xgb_grid_search = GridSearchCV(xgb_clf, xgb_params, cv=5, scoring='neg_mean_squared_error',
                               return_train_score=True, n_jobs=-1, verbose=1)

# grid search computation
xgb_joblib_file = PurePath.joinpath(model_dir, 'xgb_grid_search.sav')
xgb_grid_search_output = grid_search_func(X_strat_train, y_strat_train,  # type: ignore
                                          xgb_grid_search, xgb_joblib_file)  # type: ignore

# output list of RSME in decreasing order
grid_results(xgb_grid_search_output, prod([len(i) for i in list(xgb_params.values())]))

# accuracy performance metric
y_pred = xgb_grid_search_output.predict(X_strat_val)  # type: ignore
best_fit_acc_score = accuracy_score(y_strat_val, y_pred)  # type: ignore
print(f'Accuracy for the best-fit model: {best_fit_acc_score:.8f}')

# accuracy improvement
new_accuracy_score = best_fit_acc_score
model_clf = load(PurePath.joinpath(model_dir, '19-09-2020/02/xgb_clf.sav'))
old_accuracy_score = print_accuracy(model_clf, X_strat_val, y_strat_val)  # type: ignore
print(f'''Percentage change: {round((100*(new_accuracy_score)/old_accuracy_score)-100, 3)}%.''')

# performance metric for DrivenData competition
print(f'''Micro-averaged F1 score: {f1_score(y_strat_val, y_pred, average='micro'):.8f}''')
