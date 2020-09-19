#!/usr/bin/env python

from pandas import read_csv, concat
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from scipy.sparse import hstack
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.utils import resample


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
    train_labels_df['damage_grade'] = train_labels_df['damage_grade'].astype('category')

    return train_values_df, train_labels_df, test_values_df, num_attrib, cat_attrib


# generating training and validation data sets
def split_to_train_val_with_resampling(train_values_df, train_labels_df, upsampling=False):
    X_train, X_val, y_train, y_val = train_test_split(
        train_values_df, train_labels_df, test_size=0.3, random_state=42)
    y_train, y_val = y_train.iloc[:, 0], y_val.iloc[:, 0]
    if upsampling:
        X = hstack(X_train, y_train)
        level_list = []
        for index in range(1, 4):
            level_list.append(X[X['damage_grade'] == index])
        level_low, level_medium, level_high = level_list
        # medium level has the highest count, so let's increase the other two by upsampling
        upsampled_list = []
        for upsampled in [level_low, level_high]:
            upsampled_list.append(resample(upsampled, replace=True,  # sample with replacement
                                           n_samples=len(level_medium),  # match number in medium level
                                           random_state=33))  # reproducible results
        upsampled_list.append(level_medium)
        X_resampled = concat(upsampled_list)
        X_train = X_resampled.drop(['damage_grade'], axis=1)
        y_train = X_resampled['damage_grade']
    return X_train, X_val, y_train, y_val


# pipeline to place median for NaNs and normalize data
def feature_pipeline(train_values_df, num_attrib, cat_attrib):
    num_pipeline = Pipeline([('imputer', SimpleImputer(
        strategy='median')), ('std_scaler', StandardScaler())])
    full_pipeline = ColumnTransformer([('num', num_pipeline, num_attrib),
                                       ('cat', OneHotEncoder(), cat_attrib),
                                       ], n_jobs=-1)
    return full_pipeline.fit_transform(train_values_df)
