#!/usr/bin/env python

from pandas import read_csv, Series, concat
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from scipy.sparse import hstack, csr_matrix, vstack
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.utils import resample
from category_encoders import OneHotEncoder, TargetEncoder


# return list of dataframes, which can be tuple-unpacked
def create_dataframes(data_file_list, data_dir):
    data_frame_list = []
    for data_file in data_file_list:
        tmp_df = read_csv(data_dir / data_file, index_col='building_id')
        # tmp_df.drop('building_id', axis=1, inplace=True)
        data_frame_list.append(tmp_df)
    return data_frame_list


# convert object and binary data types to categorical data types
def prepare_data(train_values_df, test_values_df, train_labels_df):
    all_values_df = [train_values_df, test_values_df]
    other_objects = ['land_surface_condition', 'foundation_type', 'roof_type',
                     'ground_floor_type', 'other_floor_type', 'position',
                     'plan_configuration', 'legal_ownership_status']
    for df in all_values_df:
        df[other_objects] = df[other_objects].astype('category')
        filter_ = train_values_df.filter(regex='geo_level_').columns
        df[filter_] = df[filter_].astype('category')

    # list of categorical and numerical data type, necessary for ML pipeline
    # DEFINED BEFORE MERGING train_values_df AND train_labels_df
    cat_attrib = train_values_df.select_dtypes('category').columns
    num_attrib = train_values_df.select_dtypes('int64').columns

    # convert int64 numerical types to int32 to reduce memory footprint
    train_values_df[num_attrib] = train_values_df[num_attrib].astype('int32')
    test_values_df[num_attrib] = test_values_df[num_attrib].astype('int32')

    # convert object data type in train_labels_df to categorical data type
    train_labels_df['damage_grade'] = train_labels_df['damage_grade'].astype(
        'category')

    return train_values_df, train_labels_df, test_values_df, num_attrib, cat_attrib


# generating upsampled training and validation data sets from sparse matrices
def stratified_shuffle_data_split(prepared_train_values, train_labels_df):
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
    for train_index, val_index in sss.split(prepared_train_values, train_labels_df):
        X_strat_train = prepared_train_values[train_index]
        y_strat_train = train_labels_df.iloc[train_index]
        X_strat_val = prepared_train_values[val_index]
        y_strat_val = train_labels_df.iloc[val_index]
    y_strat_train, y_strat_val = y_strat_train.iloc[:, 0], y_strat_val.iloc[:, 0]  # type: ignore
    # print(f'X_strat_train.shape: {X_strat_train.shape}')  # type: ignore
    # print(f'y_strat_train: {y_strat_train.shape}')  # type: ignore
    # print(f'X_strat_val.shape: {X_strat_val.shape}')  # type: ignore
    # print(f'y_strat_val.shape: {y_strat_val.shape}\n')  # type: ignore
    return X_strat_train, y_strat_train, X_strat_val, y_strat_val  # type: ignore


# generating training and validation data sets from sparse matrices
def train_val_upsampling_split(train_values_df, train_labels_df, upsampling=False):
    X_train, X_val, y_train, y_val = train_test_split(
        train_values_df, train_labels_df, test_size=0.3, random_state=42)
    if upsampling:
        X = hstack((X_train, y_train)).toarray()
        level_list = []
        for level in range(1, 4):
            level_filter = X[:, -1] == level
            level_X = X[level_filter, 0:-1]
            level_y = X[level_filter, -1].ravel()
            level_list.append([level_X, level_y])
        level_low, level_medium, level_high = level_list
        # medium level has the highest count, so let's increase the other two by up-sampling
        upsampled_list = []
        for upsampled in [level_low, level_high]:
            tmp_upsampled_list = []
            for arr in upsampled:
                tmp_upsampled_list.append(resample(arr,
                                          replace=True,  # sample with replacement
                                          n_samples=len(level_medium[1]),  # match number in medium level
                                          random_state=33))  # reproducible results
            upsampled_list.append(tmp_upsampled_list)  # type: ignore
        upsampled_list.append(level_medium)  # notice: append works in place
        mat_x_list, mat_y_list = [], []
        for mat_x, mat_y in upsampled_list:
            mat_x_list.append(csr_matrix(mat_x))
            mat_y_list.append(mat_y)
        X_train = vstack(mat_x_list)
        y_train = Series(np.concatenate(mat_y_list, axis=0).ravel())
        y_val = y_val.iloc[:, 0]
    # print(f'X_train.shape: {X_train.shape}')
    # print(f'y_train.shape: {y_train.shape}')
    # print(f'X_val.shape: {X_val.shape}')
    # print(f'y_val.shape: {y_val.shape}\n')
    return X_train, X_val, y_train, y_val


# pipeline to place median for NaNs and normalize data
def feature_pipeline(train_values_df, num_attrib, cat_attrib):
    num_pipeline = Pipeline([('imputer', SimpleImputer(
        strategy='median')), ('std_scaler', StandardScaler())])
    full_pipeline = ColumnTransformer([('num', num_pipeline, num_attrib),
                                      #  ('cat', OneHotEncoder(), cat_attrib),
                                       ], n_jobs=-1)
    prepared_train_values = full_pipeline.fit_transform(train_values_df)
    # print(f'prepared_train_values.shape: {prepared_train_values.shape}')
    return prepared_train_values


# pipeline to place median for NaNs and normalize data
def num_feature_pipeline(train_values_df):
    num_pipeline = Pipeline([('imputer', SimpleImputer(
        strategy='median')), ('std_scaler', StandardScaler())])
    prepared_train_values = num_pipeline.fit_transform(train_values_df)
    # print(f'prepared_train_values.shape: {prepared_train_values.shape}')
    return prepared_train_values


# one-hot encode categorical columns and create mean-target encoding columns in dataframe
def target_encode_multiclass(train_values_df, train_labels_df):
    onehot_enc = OneHotEncoder()
    train_labels_onehot_df = onehot_enc.fit_transform(train_labels_df)
    class_names = train_labels_onehot_df.columns
    train_values_cat_df = train_values_df.select_dtypes('category')
    train_values_num_df = train_values_df.select_dtypes(exclude='category')
    for class_name in class_names:
        enc = TargetEncoder()
        enc.fit(train_values_cat_df, train_labels_onehot_df[class_name])
        temp_df = enc.transform(train_values_cat_df)
        temp_df.columns = [str(col)+'_'+str(class_name) for col in temp_df.columns]
        train_values_num_df = concat([train_values_num_df, temp_df], axis=1)
    return train_values_num_df
