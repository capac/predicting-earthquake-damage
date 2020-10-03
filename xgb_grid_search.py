#!/usr/bin/env python

from pathlib import Path, PurePath
from helper_funcs.clf_funcs import grid_search_func, grid_results, print_accuracy
from helper_funcs.data_preparation import create_dataframes, prepare_data, \
    stratified_shuffle_data_split
from helper_funcs.exploratory import target_encode_multiclass
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score
from joblib import load
from numpy import prod
from pandas import DataFrame
# from sklearn.ensemble import VotingClassifier

data_dir = Path('./data')
model_dir = Path('./models')

# create dataframes from csv files
data_file_list = ['train_values.csv', 'test_values.csv', 'train_labels.csv']
train_values_df, test_values_df, train_labels_df = create_dataframes(
    data_file_list, data_dir)

# convert object data types to category data types,
# numerical data types from int64 to int 32
train_values_df, train_labels_df, test_values_df, num_attrib, cat_attrib = \
    prepare_data(train_values_df, test_values_df, train_labels_df)

# pipeline to place median for NaNs and normalize data
# prepared_X_train_values = feature_pipeline(train_values_df, num_attrib, cat_attrib)

# one-hot encode categorical columns and create mean-target encoding columns in dataframe
prepared_X_train_values, prepared_test_values = \
    target_encode_multiclass(train_values_df, train_labels_df, test_values_df)

# generating stratified training and validation data sets from sparse matrices
prepared_X_strat_train, y_strat_train_df, prepared_X_strat_val, y_strat_val_df = \
    stratified_shuffle_data_split(prepared_X_train_values, train_labels_df)

# grid search setup on XGBClassifier
xgb_clf = XGBClassifier(n_jobs=-1, verbosity=1, tree_method='auto')
xgb_params = {'max_depth': [10, 15, 20], 'n_estimators': [100, 200]}
xgb_grid_search = GridSearchCV(xgb_clf, xgb_params, cv=3, scoring='f1_micro',
                               return_train_score=True, n_jobs=-1, verbose=1)

# grid search computation
xgb_joblib_file = PurePath.joinpath(model_dir, 'xgb_grid_search.sav')
xgb_grid_search_output = grid_search_func(prepared_X_strat_train, y_strat_train_df,  # type: ignore
                                          xgb_grid_search, xgb_joblib_file)  # type: ignore

# output list of RSME in decreasing order
grid_results(xgb_grid_search_output, prod([len(i) for i in list(xgb_params.values())]))

# accuracy performance metric
y_pred = xgb_grid_search_output.predict(prepared_X_strat_val)  # type: ignore
best_fit_acc_score = accuracy_score(y_strat_val_df, y_pred)  # type: ignore
print(f'Accuracy for the best-fit model: {best_fit_acc_score:.8f}')

# accuracy improvement
new_accuracy_score = best_fit_acc_score
model_clf = load(PurePath.joinpath(model_dir, '02-10-2020/01/xgb_grid_search.sav'))
old_accuracy_score = print_accuracy(model_clf, prepared_X_strat_val, y_strat_val_df)  # type: ignore
print(f'''Percentage change: {round((100*(new_accuracy_score)/old_accuracy_score)-100, 3)}%.''')

# performance metric for DrivenData competition
print(f'''Micro-averaged F1 score: {f1_score(y_strat_val_df, y_pred, average='micro'):.8f}''')

# save predicted results from test data for DrivenData competition
model_clf = load(PurePath.joinpath(model_dir, 'xgb_grid_search.sav'))
predicted_y_results = model_clf.predict(prepared_test_values)
print(f'type(predicted_y_results): {type(predicted_y_results)}')
print(f'predicted_y_results.shape: {predicted_y_results.shape}')
print(f'predicted_y_results[:10]: {predicted_y_results[:10]}')
predicted_y_results_s = DataFrame(predicted_y_results, index=test_values_df.index, columns=['damage_grade'])
predicted_y_results_s.to_csv('predicted_results.csv')
