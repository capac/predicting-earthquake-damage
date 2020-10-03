# /usr/bin/env python

from helper_funcs.data_preparation import create_dataframes, prepare_data, \
    stratified_shuffle_data_split
from helper_funcs.clf_funcs import clf_func, run_clf
from helper_funcs.exploratory import target_encode_multiclass
# from sklearn.linear_model import LogisticRegression
# from sklearn.ensemble import RandomForestClassifier
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
# prepared_X_train_values = feature_pipeline(train_values_df, num_attrib, cat_attrib)
# prepared_X_test_values = feature_pipeline(test_values_df, num_attrib, cat_attrib)

prepared_X_train_values, prepared_test_values = \
    target_encode_multiclass(train_values_df, train_labels_df, test_values_df)

# generating stratified training and validation data sets from sparse matrices
prepared_X_strat_train, y_strat_train_df, prepared_X_strat_val, y_strat_val_df = \
    stratified_shuffle_data_split(prepared_X_train_values, train_labels_df)

# classifiers employed for training
classifier_dict = {
                   'xgb_clf': XGBClassifier(tree_method='auto', n_jobs=-1, verbosity=1, max_depth=8),
                   #  'lr_clf': LogisticRegression(random_state=42, n_jobs=-1, max_iter=1e4),
                   #  'rf_clf': RandomForestClassifier(n_estimators=500),
                   'cat_clf': CatBoostClassifier(iterations=2e3,
                                                 learning_rate=0.6,
                                                 loss_function='MultiClass',
                                                 custom_metric=['Accuracy', 'AUC', 'TotalF1'],
                                                 verbose=100),
                   }

# creates list of named classifier tuples for training
clf_list = clf_func(classifier_dict)

# runs actual training on classifiers and outputs results to screen
run_clf(prepared_X_strat_train, prepared_X_strat_val, y_strat_train_df, y_strat_val_df, clf_list, model_dir)

# save predicted results from test data for DrivenData competition
model_clf = load(PurePath.joinpath(model_dir, 'nbcat_clf.sav'))
predicted_y_results = model_clf.predict(prepared_test_values)
print(f'type(predicted_y_results): {type(predicted_y_results)}')
print(f'predicted_y_results.shape: {predicted_y_results.shape}')
print(f'predicted_y_results[:10]: {predicted_y_results[:10]}')
predicted_y_results_s = DataFrame(predicted_y_results, index=test_values_df.index, columns=['damage_grade'])
predicted_y_results_s.to_csv('predicted_results.csv')
