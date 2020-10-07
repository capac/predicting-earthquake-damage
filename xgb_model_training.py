#!/usr/bin/env python

from helper_funcs.data_preparation import create_dataframes, prepare_data, \
    stratified_shuffle_data_split
from helper_funcs.clf_funcs import clf_func, run_clf
from helper_funcs.exploratory import target_encode_multiclass
from xgboost import XGBClassifier
from pathlib import Path, PurePath
from joblib import load
from pandas import DataFrame


# directory paths
data_dir = Path('./data')
model_dir = Path('./models/10-06-2020/03')

# create dataframes from csv files
data_file_list = ['train_values.csv', 'test_values.csv', 'train_labels.csv']
train_values_df, test_values_df, train_labels_df = create_dataframes(
    data_file_list, data_dir)

# convert object data types to category data types, numerical data types from int64 to int 32
train_values_df, train_labels_df, test_values_df, num_attrib, cat_attrib = \
    prepare_data(train_values_df, test_values_df, train_labels_df)

# one-hot encodes categorical columns and create mean-target encoding columns in dataframe
prepared_X_train_values, prepared_test_values = \
    target_encode_multiclass(train_values_df, train_labels_df, test_values_df)

# generating stratified training and validation data sets from sparse matrices
prepared_X_strat_train, y_strat_train_df, prepared_X_strat_val, y_strat_val_df = \
    stratified_shuffle_data_split(prepared_X_train_values, train_labels_df)

# classifiers employed for training
classifier_dict = {
                   'xgb_clf': XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
                                            colsample_bynode=1, colsample_bytree=0.8852444528883149,
                                            gamma=0.30582658024414044, gpu_id=-1, importance_type='gain',
                                            interaction_constraints='', learning_rate=0.034239783131830445,
                                            max_delta_step=0, max_depth=13, min_child_weight=1, missing=None,
                                            monotone_constraints='()', n_estimators=148, n_jobs=-1,
                                            num_parallel_tree=1, objective='multi:softprob', random_state=0,
                                            reg_alpha=0, reg_lambda=1, scale_pos_weight=None,
                                            subsample=0.9099098641033556, tree_method='auto',
                                            validate_parameters=1, verbosity=1),
                   }

# creates list of named classifier tuples for training
clf_list = clf_func(classifier_dict)

# runs actual training on classifiers and outputs results to screen
run_clf(prepared_X_strat_train, prepared_X_strat_val, y_strat_train_df, y_strat_val_df, clf_list, model_dir)

# save predicted results from test data for DrivenData competition
model_clf = load(PurePath.joinpath(model_dir, 'xgb_clf.sav'))
predicted_y_results = model_clf.predict(prepared_test_values)
print(f'type(predicted_y_results): {type(predicted_y_results)}')
print(f'predicted_y_results.shape: {predicted_y_results.shape}')
print(f'predicted_y_results[:10]: {predicted_y_results[:10]}')
predicted_y_results_s = DataFrame(predicted_y_results, index=test_values_df.index, columns=['damage_grade'])
predicted_y_results_s.to_csv(PurePath.joinpath(model_dir, 'predicted_results.csv'))
