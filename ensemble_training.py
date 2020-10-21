#!/usr/bin/env python

from helper_funcs.data_preparation import create_dataframes, prepare_data, \
    stratified_shuffle_data_split
from helper_funcs.clf_funcs import run_ensemble_clf
from helper_funcs.exploratory import target_encode_multiclass
from sklearn.ensemble import VotingClassifier
# from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
# from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from pathlib import Path, PurePath
from joblib import load
from pandas import DataFrame


# directory paths
data_dir = Path('./data')
model_dir = Path('./models')

# create dataframes from csv files
data_file_list = ['train_values.csv', 'test_values.csv', 'train_labels.csv']
train_values_df, test_values_df, train_labels_df = create_dataframes(
    data_file_list, data_dir)

# convert object data types to category data types, numerical data types from int64 to int 32
train_values_df, train_labels_df, test_values_df, num_attrib, cat_attrib = \
    prepare_data(train_values_df, test_values_df, train_labels_df)

# pipeline to place median for NaNs and normalize data
prepared_X_train_values, prepared_test_values = \
    target_encode_multiclass(train_values_df, train_labels_df, test_values_df)

# generating stratified training and validation data sets from sparse matrices
prepared_X_strat_train, y_strat_train_df, prepared_X_strat_val, y_strat_val_df = \
    stratified_shuffle_data_split(prepared_X_train_values, train_labels_df)

# classifiers employed for training
classifier_dict = {
                   # removes 'catinfo' directory with model training logs
                   #  'cat_clf': CatBoostClassifier(allow_writing_files=False, early_stopping_rounds=20),
                   'cat_clf': CatBoostClassifier(allow_writing_files=False, verbose=100),
                   #  'xgb_clf': XGBClassifier(early_stopping_rounds=20, verbosity=0),
                   'xgb_clf': XGBClassifier(verbosity=0),
                   # From the SGDClassifier docs: "'modified_huber' is another
                   # smooth loss that brings tolerance to outliers as well as
                   # probability estimates", which is required with voting='soft'
                   # in VotingClassifier.
                   'sgd_clf': SGDClassifier(loss='modified_huber'),
                   #  'lr_clf': OneVsRestClassifier(LogisticRegression(max_iter=1000)),
                   'rf_clf': RandomForestClassifier(),
                   'ada_clf': AdaBoostClassifier()
                   }

# creates list of named classifier tuples for training
clf_list = [(key, val) for key, val in classifier_dict.items()]

# using voting='soft' from https://github.com/scikit-learn/scikit-learn/issues/15368
voting_clf = VotingClassifier(estimators=clf_list, voting='soft', n_jobs=-1, verbose=True)

# runs actual training on classifiers and outputs results to screen
run_ensemble_clf(prepared_X_strat_train, prepared_X_strat_val, y_strat_train_df, y_strat_val_df, voting_clf, model_dir)

# pure joblib load doesn't work, use open to serialize as raw binary object
with open(PurePath.joinpath(model_dir, 'voting_clf.sav'), 'rb') as f:
    model_clf = load(f)
predicted_y_results = model_clf.predict(prepared_test_values)
print(f'\ntype(predicted_y_results): {type(predicted_y_results)}')
print(f'predicted_y_results.shape: {predicted_y_results.shape}')
print(f'predicted_y_results[:10]: {predicted_y_results[:10]}')

# save predicted results from test data for DrivenData competition
predicted_y_results_df = DataFrame(predicted_y_results, index=test_values_df.index, columns=['damage_grade'])
predicted_y_results_df.to_csv('predicted_results.csv')
print('Done!')
