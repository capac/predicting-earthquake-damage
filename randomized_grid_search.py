#!/usr/bin/env python

from pathlib import Path, PurePath
from helper_funcs.funcs import grid_search_func, grid_results, print_accuracy
from helper_funcs.data_preparation import create_dataframes, prepare_data, \
    feature_pipeline, stratified_shuffle_data_split
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform, randint
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score
from joblib import load
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
prepared_train_values = feature_pipeline(train_values_df, num_attrib, cat_attrib)

# generating stratified training and validation data sets from sparse matrices
X_strat_train, y_strat_train, X_strat_val, y_strat_val = \
    stratified_shuffle_data_split(prepared_train_values, train_labels_df)

# grid search setup on XGBClassifier
xgb_clf = XGBClassifier(n_jobs=-1, verbosity=1, tree_method='hist')
# xgb_params = {'max_depth': [10, 15, 20], 'n_estimators': [100, 200]}

xgb_params = {'colsample_bytree': uniform(0.7, 0.3),
              'gamma': uniform(0, 0.5),
              'learning_rate': uniform(0.003, 0.3),
              'max_depth': randint(5, 20),  # xgb_grid_search best fit parameter: 15
              'n_estimators': randint(100, 200),  # xgb_grid_search best fit parameter: 200
              'subsample': uniform(0.6, 0.4)}

xgb_grid_search = RandomizedSearchCV(xgb_clf, param_distributions=xgb_params, random_state=42,
                                     n_iter=10, cv=5, verbose=1, n_jobs=-1, return_train_score=True,
                                     scoring='f1_micro')

# xgb_grid_search = GridSearchCV(xgb_clf, xgb_params, cv=3, scoring='neg_mean_squared_error',
#                                return_train_score=True, n_jobs=-1, verbose=1)

# grid search computation
xgb_joblib_file = PurePath.joinpath(model_dir, 'xgb_grid_search.sav')
xgb_grid_search_output = grid_search_func(X_strat_train, y_strat_train,  # type: ignore
                                          xgb_grid_search, xgb_joblib_file)  # type: ignore

# output list of RSME in decreasing order
grid_results(xgb_grid_search_output, 8)

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
