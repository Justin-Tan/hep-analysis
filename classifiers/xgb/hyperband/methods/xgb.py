#!/usr/bin/env python

# xgb methods required for HyperBand algorithm

import pandas as pd
import xgboost as xgb
import sys, time, os

from hyperopt import hp
import hyperopt.pyll.stochastic
from hyperband import Hyperband
from pprint import pprint

def xgb_hyp():
    gbtree_hyp = {
        'booster': 'gbtree',
        'eta': hp.uniform('lr', 0.01, 0.15),
        'gamma': hp.uniform('mlr', 0.05, 2.5),
        'min_child_weight': hp.uniform('mcw', 0, 2),
        'max_depth': hp.quniform('md', 3, 9, 1),
        'subsample': hp.uniform('ss', 0.7, 1),
        'colsample_bytree': hp.uniform('cs', 0.7, 1),
        'objective': 'binary:logistic',
        'silent': 1
    }
    
    dart_hyp = {
        'booster': 'dart',
        'sample_type': hp.choice('dart_st', ['uniform', 'weighted']),
        'normalize_type': hp.choice('dart_nt', ['tree', 'forest']),
        'rate_drop': hp.uniform('dropout', 0, 0.3),
        'skip_drop': hp.uniform('skip', 0, 0.25)
    }
    
    return gbtree_hyp, dart_hyp

def load_data(fname, test_size = 0.05):
    from sklearn.model_selection import train_test_split
    df = pd.read_hdf(fname, 'df')
    # Split data into training, testing sets
    df_X_train, df_X_test, df_y_train, df_y_test = train_test_split(df.drop(['labels', 'mbc', 'deltae'], axis = 1),
            df['labels'], test_size = test_size, random_state=42)

    dTrain = xgb.DMatrix(data = df_X_train.values, label = df_y_train.values, feature_names = df_X_train.columns)
    dTest = xgb.DMatrix(data = df_X_test.values, label = df_y_test.values, feature_names = df_X_test.columns)

    print('# Features: {} | # Train Samples: {} | # Test Samples: {}'.format(dTrain.num_col(),
     dTrain.num_row(), dTest.num_row()))

    # Save to XGBoost binary file for faster loading
    dTrain.save_binary("dTrain.buffer")
    dTest.save_binary("dTest.buffer")

    return dTrain, dTest

def load_data_exclusive(fname, mode, channel, test_size = 0.05):
    from sklearn.model_selection import train_test_split
    # Split data into training, testing sets
    df = pd.read_hdf(fname, 'df')
    df_X_train, df_X_test, df_y_train, df_y_test = train_test_split(df[df.columns[:-1]], df['labels'],
                                                          test_size = test_size, random_state = 24601)

    dTrain = xgb.DMatrix(data = df_X_train.values, label = df_y_train.values, feature_names = df.columns[:-1])
    dTest = xgb.DMatrix(data = df_X_test.values, label = df_y_test.values, feature_names = df.columns[:-1])
    # Save to XGBoost binary file for faster loading
    dTrain.save_binary("dTrain" + mode + channel + ".buffer")
    dTest.save_binary("dTest" + mode + channel + ".buffer")
    
    return dTrain, dTest


def train_hyp_config(data, hyp_params, num_boost_rounds):
    # Returns validation metric after training configuration for allocated resources
    # Inputs: data - DMatrix tuple: (train, test)

    # Add evaluation metrics for validation set
    hyp_params['eval_metric'] = 'error@0.5'
    pList = list(hyp_params.items())+[('eval_metric', 'auc')]
    
    # Number of boosted trees to construct
    nTrees = num_boost_rounds
    # Specify validation set to watch performance
    dTrain, dTest = data[0], data[1]
    evalList  = [(dTrain,'train'), (dTest,'eval')]

    print("Starting model training\n")
    start_time = time.time()
    # Train the model using the above parameters
    bst = xgb.train(params = pList, dtrain = dTrain, evals = evalList, num_boost_round = nTrees, 
                    early_stopping_rounds = 256, verbose_eval = int(min(64, num_boost_rounds/2)))

    delta_t = time.time() - start_time
    print("Training ended. Elapsed time: (%.3f s)." %(delta_t))
    pprint(bst.attributes())
    
    evalDict = {'auc': float(bst.attr('best_score')), 'error@0.5': bst.attr('best_msg').split('\t')[-2],
                'best_iteration': int(bst.attr('best_iteration'))}
    
    return evalDict


def get_hyp_config():
    # Returns a set of i.i.d samples from a distribution over the hyperparameter space
    gbtree_hyp, dart_hyp = xgb_hyp()
    space = hp.choice('booster', [gbtree_hyp, {**gbtree_hyp, **dart_hyp}])
    params = hyperopt.pyll.stochastic.sample(space)
    for k, v in params.items():
        if type(v) == float and int(v) == v:
            params[k] = int(v)
    params = {k: v for k, v in params.items() if v is not 'default'}
    return params


def run_hyp_config(data, hyp_params, n_iterations, rounds_per_iteration = 64):
    """
    Input: Training data, Hyperparameter configuration (t); resource allocation (r)
    Returns: Validation metric after training configuration for allocated resources
    """
    num_boost_rounds = int(round(n_iterations*rounds_per_iteration))
    print("Boosting iterations: %d"%(num_boost_rounds))
    pprint(hyp_params)
    
    return train_hyp_config(data, hyp_params, num_boost_rounds)
