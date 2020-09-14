#!/usr/bin/env python

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer


def prepare_data(train_values_df, test_values_df, train_labels_df):
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
    return train_values_df, test_values_df, train_labels_df, num_attrib, cat_attrib


def feature_pipeline(train_values_df, num_attrib, cat_attrib):
    # pipeline to place median for NaNs and normalize data
    num_pipeline = Pipeline([('imputer', SimpleImputer(strategy='median')), ('std_scaler', StandardScaler())])
    full_pipeline = ColumnTransformer([('num', num_pipeline, num_attrib), ('cat', OneHotEncoder(), cat_attrib)])
    train_values_prepared_df = full_pipeline.fit_transform(train_values_df)
    return train_values_prepared_df
