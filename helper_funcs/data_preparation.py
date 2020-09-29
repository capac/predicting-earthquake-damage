#!/usr/bin/env python

from pandas import read_csv
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from category_encoders.binary import BinaryEncoder
# from category_encoders.ordinal import OrdinalEncoder
# from sklearn.preprocessing import OneHotEncoder


# return list of dataframes, which can be tuple-unpacked
def create_dataframes(data_file_list, data_dir):
    data_frame_list = []
    for data_file in data_file_list:
        tmp_df = read_csv(data_dir / data_file, index_col='building_id')
        # tmp_df.drop('building_id', axis=1, inplace=True)
        tmp_df.name = str(data_file).split('.')[0]
        data_frame_list.append(tmp_df)
    return data_frame_list


# convert object and binary data types to categorical data types
def prepare_data(train_values_df, test_values_df, train_labels_df):
    all_values_df = [train_values_df, test_values_df]
    other_objects = ['land_surface_condition', 'foundation_type', 'roof_type',
                     'ground_floor_type', 'other_floor_type', 'position',
                     'plan_configuration', 'legal_ownership_status']
    # geo_list = ['geo_level_2_id', 'geo_level_3_id']  # testing on subset of geographic regions
    for df in all_values_df:
        df[other_objects] = df[other_objects].astype('category')
        filter_ = train_values_df.filter(regex='geo_level_').columns
        df[filter_] = df[filter_].astype('category')
        # df.drop(geo_list, axis=1, inplace=True)

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

    # print datframe shapes
    print(f'train_values_df.shape: {train_values_df.shape}')
    print(f'train_labels_df.shape: {train_labels_df.shape}')
    print(f'test_values_df.shape: {test_values_df.shape}\n')
    return train_values_df, train_labels_df, test_values_df, num_attrib, cat_attrib


# pipeline to place median for NaNs and normalize data
def feature_pipeline(train_values_df, num_attrib, cat_attrib):
    num_pipeline = Pipeline([('imputer', SimpleImputer(
        strategy='median')), ('std_scaler', StandardScaler())])
    full_pipeline = ColumnTransformer([('num', num_pipeline, num_attrib),
                                       ('cat', BinaryEncoder(), cat_attrib),
                                       ], n_jobs=-1)
    prepared_train_values = full_pipeline.fit_transform(train_values_df)
    # print datframe shapes
    print(f'''prepared_{train_values_df.name}.shape: {prepared_train_values.shape}''')
    print(f'''type(prepared_{train_values_df.name}): {type(prepared_train_values)}\n''')
    return prepared_train_values


# generating upsampled training and validation data sets from sparse matrices
def stratified_shuffle_data_split(prepared_train_values, train_labels_df):
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
    for train_index, val_index in sss.split(prepared_train_values, train_labels_df):
        prepared_X_strat_train = prepared_train_values[train_index]
        y_strat_train_df = train_labels_df.iloc[train_index]
        prepared_X_strat_val = prepared_train_values[val_index]
        y_strat_val_df = train_labels_df.iloc[val_index]
    y_strat_train_df, y_strat_val_df = y_strat_train_df.iloc[:, 0], y_strat_val_df.iloc[:, 0]  # type: ignore
    print(f'prepared_X_strat_train.shape: {prepared_X_strat_train.shape}')  # type: ignore
    print(f'type(prepared_X_strat_train): {type(prepared_X_strat_train)}')  # type: ignore
    print(f'prepared_X_strat_val.shape: {prepared_X_strat_val.shape}')  # type: ignore
    print(f'type(prepared_X_strat_val): {type(prepared_X_strat_val)}\n')  # type: ignore

    # show stratified sampling according to initial proportion of damage levels in training labels
    print('y_strat_train_df.value_counts()/len(y_strat_train_df):')
    print(f'{y_strat_train_df.value_counts()/len(y_strat_train_df)}\n')  # type: ignore
    print('y_strat_val_df.value_counts()/len(y_strat_val_df):')
    print(f'{y_strat_val_df.value_counts()/len(y_strat_val_df)}\n')  # type: ignore

    return prepared_X_strat_train, y_strat_train_df, prepared_X_strat_val, y_strat_val_df  # type: ignore
