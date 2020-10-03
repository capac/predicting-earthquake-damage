#!/usr/bin/env python

from pathlib import Path, PurePath
from helper_funcs.clf_funcs import grid_search_func, grid_results, print_accuracy
from helper_funcs.data_preparation import create_dataframes, prepare_data, feature_pipeline, \
    stratified_shuffle_data_split
from sklearn.model_selection import GridSearchCV
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, f1_score
from joblib import load
from numpy import prod
from pandas import DataFrame

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
prepared_X_train_values = feature_pipeline(train_values_df, num_attrib, cat_attrib)
prepared_X_test_values = feature_pipeline(test_values_df, num_attrib, cat_attrib)

# generating stratified training and validation data sets from sparse matrices
prepared_X_strat_train, y_strat_train_df, prepared_X_strat_val, y_strat_val_df = \
    stratified_shuffle_data_split(prepared_X_train_values, train_labels_df)

# grid search setup on XGBClassifier
cat_clf = CatBoostClassifier()
cat_params = {'iterations': [2e3, 3e3], 'learning_rate': [0.4, 0.6, 0.8]}
cat_grid_search = GridSearchCV(cat_clf, cat_params, cv=3, scoring='f1_micro',
                               return_train_score=True, n_jobs=-1, verbose=100)

# grid search computation
cat_joblib_file = PurePath.joinpath(model_dir, 'cat_grid_search.sav')
cat_grid_search_output = grid_search_func(prepared_X_strat_train, y_strat_train_df,  # type: ignore
                                          cat_grid_search, cat_joblib_file)  # type: ignore

# output list of RSME in decreasing order
grid_results(cat_grid_search_output, prod([len(i) for i in list(cat_params.values())]))

# accuracy performance metric
y_pred = cat_grid_search_output.predict(prepared_X_strat_val)  # type: ignore
best_fit_acc_score = accuracy_score(y_strat_val_df, y_pred)  # type: ignore
print(f'Accuracy for the best-fit model: {best_fit_acc_score:.8f}')

# accuracy improvement
new_accuracy_score = best_fit_acc_score
model_clf = load(PurePath.joinpath(model_dir, '01-10-2020/03/cat_clf.sav'))
old_accuracy_score = print_accuracy(model_clf, prepared_X_strat_val, y_strat_val_df)  # type: ignore
print(f'''Percentage change: {round((100*(new_accuracy_score)/old_accuracy_score)-100, 3)}%.''')

# performance metric for DrivenData competition
print(f'''Micro-averaged F1 score: {f1_score(y_strat_val_df, y_pred, average='micro'):.8f}''')

# save predicted results from test data for DrivenData competition
model_clf = load(PurePath.joinpath(model_dir, '01-10-2020/05/cat_clf.sav'))
predicted_y_results = model_clf.predict(prepared_X_test_values)
print(f'type(predicted_y_results): {type(predicted_y_results)}')
print(f'predicted_y_results.shape: {predicted_y_results.shape}')
print(f'predicted_y_results[:10]: {predicted_y_results[:10]}')
predicted_y_results_s = DataFrame(predicted_y_results, index=test_values_df.index, columns=['damage_grade'])
predicted_y_results_s.to_csv('predicted_results.csv')
