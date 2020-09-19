# /usr/bin/env python

from helper_funcs.data_preparation import create_dataframes, prepare_data, \
    feature_pipeline
from helper_funcs.funcs import clf_func, run_clf
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from pathlib import Path


# directory paths
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

# generating stratified training and validation data sets
# sss = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
# for train_index, val_index in sss.split(train_values_df, train_values_df['damage_grade']):
#     strat_train_set = train_values_df.loc[train_index]
#     strat_val_set = train_values_df.loc[val_index]

# pipeline to place median for NaNs and normalize data
prepared_train_values_df = feature_pipeline(
    train_values_df, num_attrib, cat_attrib)

print(f'prepared_train_values_df.shape: {prepared_train_values_df.shape}')

X_train, X_val, y_train, y_val = train_test_split(prepared_train_values_df, train_labels_df,
                                                  test_size=0.3, random_state=42)
y_train, y_val = y_train.iloc[:, 0], y_val.iloc[:, 0]

print(f'X_train.shape: {X_train.shape}')
print(f'y_train.shape: {y_train.shape}')
print(f'X_val.shape: {X_val.shape}')
print(f'y_val.shape: {y_val.shape}\n')

print(f'y_train.value_counts()/len(y_train):\n{y_train.value_counts()/len(y_train)}\n')
print(f'train_labels_df.value_counts()/len(train_labels_df):\n{train_labels_df.value_counts()/len(train_labels_df)}\n')

# classifiers employed for training
classifier_dict = {'lr_clf': LogisticRegression(random_state=42, n_jobs=-1, max_iter=1e4),
                   'xgb_clf': XGBClassifier(n_jobs=-1, verbosity=1, max_depth=8, tree_method='auto'),
                   }

# creates list of named classifier tuples for training
clf_list = clf_func(classifier_dict)

# print(f'num_attrib: {num_attrib}\n')
# print(f'cat_attrib: {cat_attrib}')

# print(f'type(prepared_X_train): {type(prepared_X_train)}')
# print(f'prepared_X_train.toarray().shape:\n{prepared_X_train.toarray().shape}\n')

# runs actual training on classifiers and outputs results to screen
run_clf(X_train, X_val, y_train, y_val, clf_list, model_dir)
