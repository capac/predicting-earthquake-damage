#! /usr/bin/env python

import category_encoders as ce
from pandas import concat, Series
from time import time
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from scipy.sparse import hstack, csr_matrix, vstack
import numpy as np
from sklearn.utils import resample


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
    print(f'X_train.shape: {X_train.shape}')
    print(f'y_train.shape: {y_train.shape}')
    print(f'X_val.shape: {X_val.shape}')
    print(f'y_val.shape: {y_val.shape}\n')
    return X_train, X_val, y_train, y_val


# one-hot encodes categorical columns and create mean-target encoding columns in dataframe
def target_encode_multiclass(train_values_df, train_labels_df, test_values_df=None):
    onehot_enc = ce.OneHotEncoder()
    train_labels_onehot_df = onehot_enc.fit_transform(train_labels_df)
    class_names = train_labels_onehot_df.columns
    cat_train_values_df = train_values_df.select_dtypes('category')
    num_train_values_df = train_values_df.select_dtypes(exclude='category')
    num_pipeline = Pipeline([('imputer', SimpleImputer(strategy='median')),
                             ('std_scaler', StandardScaler())])
    num_train_values_df = num_pipeline.fit_transform(num_train_values_df)
    if test_values_df:
        cat_test_values_df = test_values_df.select_dtypes('category')
        num_test_values_df = test_values_df.select_dtypes(exclude='category')
    for class_name in class_names:
        enc = ce.TargetEncoder()
        enc.fit(cat_train_values_df, train_labels_onehot_df[class_name])
        temp_train_df = enc.transform(cat_train_values_df)
        temp_train_df.columns = [str(col)+'_'+str(class_name) for col in temp_train_df.columns]
        num_train_values_df = concat([num_train_values_df, temp_train_df], axis=1)
        if test_values_df:
            temp_test_df = enc.transform(cat_test_values_df)  # type: ignore
            temp_test_df.columns = [str(col)+'_'+str(class_name) for col in temp_test_df.columns]
            num_test_values_df = concat([num_test_values_df, temp_test_df], axis=1)  # type: ignore
    if test_values_df:
        return num_train_values_df, num_test_values_df  # type: ignore
    else:
        return num_train_values_df


# multi-category encoder function to test category functions in category_encoders with earthquake data
def test_category_encoders(classifier, X_strat_train, y_strat_train, X_strat_val, y_strat_val, num_attrib, cat_attrib):
    encoder_list = [
                    ce.backward_difference.BackwardDifferenceEncoder,
                    ce.basen.BaseNEncoder,
                    ce.binary.BinaryEncoder,
                    ce.cat_boost.CatBoostEncoder,  # didn't work with earthquake data
                    ce.hashing.HashingEncoder,  # didn't work with earthquake data
                    ce.helmert.HelmertEncoder,
                    ce.james_stein.JamesSteinEncoder,
                    ce.one_hot.OneHotEncoder,  # didn't work with earthquake data
                    ce.leave_one_out.LeaveOneOutEncoder,  # didn't work with earthquake data
                    ce.m_estimate.MEstimateEncoder,
                    ce.ordinal.OrdinalEncoder,
                    ce.polynomial.PolynomialEncoder,
                    ce.sum_coding.SumEncoder,
                    ce.target_encoder.TargetEncoder,  # didn't work with earthquake data
                    ce.woe.WOEEncoder
                    ]
    for encoder in encoder_list:
        t0 = time()
        numeric_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='median')),
                                              ('scaler', StandardScaler())])
        categorical_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='constant',
                                                  fill_value='missing')), ('woe', encoder())])

        preprocessor = ColumnTransformer(transformers=[('num', numeric_transformer, num_attrib),
                                                       ('cat', categorical_transformer, cat_attrib)], n_jobs=-1)

        pipe = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', classifier)])
        print(f'Fitting encoder: {encoder}')
        model = pipe.fit(X_strat_train, y_strat_train)

        y_pred = model.predict(X_strat_val)
        print(f'''Time elapsed: {time() - t0:.4f} sec, micro-averaged F1 score: {f1_score(y_strat_val,y_pred,
        average='micro'):.8f}''')
