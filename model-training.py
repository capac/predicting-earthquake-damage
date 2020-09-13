#/usr/bin/env python

import pandas as pd
from pathlib import Path, PurePath
from time import time
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
# from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from xgboost import XGBClassifier
from joblib import dump, load
from helper_funcs.funcs import clf_func, print_accuracy

data_dir = Path('./data')
model_dir = Path('./models')

# data frame creation
data_frame_list = []
data_file_list = ['train_values.csv', 'test_values.csv', 'train_labels.csv']
for data_file in data_file_list:
    data_frame_list.append(pd.read_csv(data_dir / data_file, index_col='building_id'))
train_values_df, test_values_df, train_labels_df = data_frame_list

# convert object data types in train_values_df, test_values_df to category data types
all_dataframes = [train_values_df, test_values_df]
other_objects = ['land_surface_condition', 'foundation_type', 'roof_type', 'ground_floor_type', \
                 'other_floor_type', 'position',  'plan_configuration', 'legal_ownership_status']
all_regexes = ['has_', 'geo_level_']
for df in all_dataframes:
    df[other_objects] = df[other_objects].astype('category')
    for reg in all_regexes:
        filter_ = train_values_df.filter(regex=reg).columns
        df[filter_] = df[filter_].astype('category')

# list of categorical data type, necessary for pipeline
cat_attrib = train_values_df.select_dtypes('category').columns

# convert numerical types in train_values_df, test_values_df from int64 to int32 to reduce memory use
num_attrib = train_values_df.select_dtypes('int64').columns
train_values_df[num_attrib] = train_values_df[num_attrib].astype('int32')
test_values_df[num_attrib] = test_values_df[num_attrib].astype('int32')

# convert object data type in train_labels_df to category type
train_labels_df['damage_grade'] = train_labels_df['damage_grade'].astype('category')

# pipeline to place median for NaNs and normalize data
num_pipeline = Pipeline([('imputer', SimpleImputer(strategy='median')), ('std_scaler', StandardScaler())])
full_pipeline = ColumnTransformer([('num', num_pipeline, num_attrib), ('cat', OneHotEncoder(), cat_attrib)])
train_values_prepared_df = full_pipeline.fit_transform(train_values_df)

# generating training and test data sets
X_train, X_test, y_train, y_test = train_test_split(train_values_prepared_df, train_labels_df, test_size = 0.3, random_state=42)
y_train, y_test = y_train.iloc[:,0], y_test.iloc[:,0]

classifier_dict = {'xgb_clf': XGBClassifier(n_jobs=-1, verbosity=1, max_depth=24, tree_method='auto'),
                   'lin_svc_clf': LinearSVC(dual=False, random_state=42), 
                #    'rf_clf': RandomForestClassifier(random_state=42, n_jobs=-1),
                   'lr_clf': LogisticRegression(random_state=42, n_jobs=-1, max_iter=1e4)
                   }

clf_list = clf_func(classifier_dict)

t0 = time()
list_files = [x for x in model_dir.iterdir() if x.is_file]
accuracy_score_dict = dict()
for item in clf_list:
    model_file = PurePath.joinpath(model_dir, item.name+'.sav')
    if model_file in list_files:
        t1 = time()
        model_clf = load(model_file)
        accuracy_score_dict[model_clf.__class__.__name__] = print_accuracy(model_clf, t1, X_test, y_test.values)
    else:
        t2 = time()
        item.classifier.fit(X_train, y_train.values)
        dump(item.classifier, model_file)
        accuracy_score_dict[item.classifier.__class__.__name__] = print_accuracy(item.classifier, t2, X_test, y_test.values)
print(f'\nTotal time elasped: {time() - t0:.4f} sec')
