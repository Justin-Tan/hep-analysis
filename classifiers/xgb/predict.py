#!/usr/bin/env python
# Quick script to make predictions with saved model

import pandas as pd
import xgboost as xgb
import sys, time, os, json
import argparse

def load_data(fname, mode, channel, test_size = 0.05, predict = False):
    # Input data as HDF5 file with labels in the last column
    from sklearn.model_selection import train_test_split
    df = pd.read_hdf(fname, 'df')
    # Split data into training, testing sets
    df_X_train, df_X_test, df_y_train, df_y_test = train_test_split(df[df.columns[:-1]], df['labels'],
                                                          test_size = test_size, random_state = 24601)

    dTrain = xgb.DMatrix(data = df_X_train.values, label = df_y_train.values, feature_names = df.columns[:-1])
    dTest = xgb.DMatrix(data = df_X_test.values, label = df_y_test.values, feature_names = df.columns[:-1])
    # Save to XGBoost binary file for faster loading
    dTrain.save_binary(os.path.join('dmatrices', "dTrain" + mode + channel + ".buffer"))
    dTest.save_binary(os.path.join('dmatrices', "dTest" + mode + channel + ".buffer"))

    return dTrain, dTest


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data_file', help = 'Path to dataset in XGBoost binary file format')
    parser.add_argument('xgb_model', help = 'Path to saved XGB model')
    parser.add_argument('channel', help = 'Decay channel')
    parser.add_argument('mode', help = 'Background type')
    parser.add_argument('-df', '--df_format', help = 'Load dataframe from HDF5 format', action = 'store_true')
    args = parser.parse_args()

    # Load saved data 
    print('Loading prediction dataset from: {}'.format(args.data_file))
    predDMatrix = xgb.Dmatrix(args.data_file)

    # Load saved model, make predictions
    bst = xgb.Booster({'nthread':4})
    bst.load_model(args.xgb_model)
    y_pred = bst.predict(predDMatrix,ntree_limit=bst.best_ntree_limit)

