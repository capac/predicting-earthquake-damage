# /usr/bin/env python

from helper_funcs.data_preparation import create_dataframes, prepare_data, \
    stratified_shuffle_data_split, feature_pipeline
from helper_funcs.funcs import clf_func, run_clf
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from pathlib import Path  # PurePath
# from joblib import load


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
prepared_X_train_values = feature_pipeline(train_values_df, num_attrib, cat_attrib)
prepared_X_test_values = feature_pipeline(test_values_df, num_attrib, cat_attrib)

# generating stratified training and validation data sets from sparse matrices
prepared_X_strat_train, y_strat_train_df, prepared_X_strat_val, y_strat_val_df = \
    stratified_shuffle_data_split(prepared_X_train_values, train_labels_df)

# classifiers employed for training
classifier_dict = {'xgb_clf': XGBClassifier(tree_method='auto', n_jobs=-1, verbosity=1, max_depth=8),
                   'lr_clf': LogisticRegression(random_state=42, n_jobs=-1, max_iter=1e4),
                   'rf_clf': RandomForestClassifier(n_estimators=500),
                   }

# creates list of named classifier tuples for training
clf_list = clf_func(classifier_dict)

# runs actual training on classifiers and outputs results to screen
run_clf(prepared_X_strat_train, prepared_X_strat_val, y_strat_train_df, y_strat_val_df, clf_list, model_dir)

# save predicted results from test data for DrivenData competition
# model_clf = load(PurePath.joinpath(model_dir, '23-09-2020/01/xgb_grid_search.sav'))
# predicted_y_results = model_clf.predict(prepared_test_values)
# print(f'type(predicted_y_results): {type(predicted_y_results)}')
