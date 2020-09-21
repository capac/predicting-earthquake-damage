# /usr/bin/env python

from helper_funcs.data_preparation import create_dataframes, prepare_data, \
    feature_pipeline, train_val_upsampling_split, stratified_shuffle_data_split
from helper_funcs.funcs import clf_func, run_clf
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
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

# pipeline to place median for NaNs and normalize data
prepared_train_values = feature_pipeline(train_values_df, num_attrib, cat_attrib)

# generating stratified training and validation data sets from sparse matrices
X_strat_train, y_strat_train, X_strat_val, y_strat_val = \
    stratified_shuffle_data_split(prepared_train_values, train_labels_df)

print(f'y_strat_train.value_counts()/len(y_strat_train):\n\
{y_strat_train.value_counts()/len(y_strat_train)}\n')  # type: ignore
print(f'y_strat_val.value_counts()/len(y_strat_val):\n\
{y_strat_val.value_counts()/len(y_strat_val)}\n')  # type: ignore

# generating up-sampled training and validation data sets from sparse matrices
X_train, X_val, y_train, y_val = train_val_upsampling_split(
    prepared_train_values, train_labels_df, upsampling=True)

print(f'y_train.value_counts()/len(y_train):\n\
{y_train.value_counts()/len(y_train)}\n')
print(f'train_labels_df.value_counts()/len(train_labels_df):\n\
{train_labels_df.value_counts()/len(train_labels_df)}\n')

# classifiers employed for training
classifier_dict = {'lr_clf': LogisticRegression(random_state=42, n_jobs=-1, max_iter=1e4),
                   'xgb_clf': XGBClassifier(n_jobs=-1, verbosity=1, max_depth=8, tree_method='auto'),
                   }

# creates list of named classifier tuples for training
clf_list = clf_func(classifier_dict)

# runs actual training on classifiers and outputs results to screen
run_clf(X_train, X_val, y_train, y_val, clf_list, model_dir)
# run_clf(X_strat_train, X_strat_val, y_strat_train, y_strat_val, clf_list, model_dir)  # type: ignore
