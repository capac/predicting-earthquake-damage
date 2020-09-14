#/usr/bin/env python

from pathlib import Path
import pandas as pd
from helper_funcs.data_preparation import prepare_data, feature_pipeline
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from helper_funcs.funcs import clf_func, print_accuracy, run_clf

# directory paths
data_dir = Path('./data')
model_dir = Path('./models')

# data frame creation
data_frame_list = []
data_file_list = ['train_values.csv', 'test_values.csv', 'train_labels.csv']
for data_file in data_file_list:
    data_frame_list.append(pd.read_csv(data_dir / data_file, index_col='building_id'))
train_values_df, test_values_df, train_labels_df = data_frame_list

# convert object and numerical data types in train_values_df, test_values_df to category data types
train_values_df, test_values_df, train_labels_df, num_attrib, cat_attrib = prepare_data(train_values_df, test_values_df, train_labels_df)

# pipeline to place median for NaNs and normalize data
train_values_prepared_df = feature_pipeline(train_values_df, num_attrib, cat_attrib)

# generating training and test data sets
X_train, X_test, y_train, y_test = train_test_split(train_values_prepared_df, train_labels_df, test_size = 0.3, random_state=42)
y_train, y_test = y_train.iloc[:,0], y_test.iloc[:,0]

classifier_dict = {'xgb_clf': XGBClassifier(n_jobs=-1, verbosity=1, max_depth=24, tree_method='auto'),
                   'lr_clf': LogisticRegression(random_state=42, n_jobs=-1, max_iter=1e4),
                   }

clf_list = clf_func(classifier_dict)
run_clf(X_train, X_test, y_train, y_test, clf_list, model_dir)
