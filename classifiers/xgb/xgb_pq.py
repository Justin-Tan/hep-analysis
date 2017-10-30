#!/usr/bin/env python
# XGBoost training script
# Optional arguments: randomly select hyperparameters

import numpy as np
import pandas as pd
import xgboost as xgb
import sys, time, os, json
import argparse
import pyarrow.parquet as pq

from hyperopt import hp
import hyperopt.pyll.stochastic
from pprint import pprint

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

def xgb_hyp():
    gbtree_hyp = {
        'booster': 'gbtree',
        'eta': hp.uniform('lr', 0.01, 0.17),
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

def hp_default_config(deep):
    if deep:
        print('Using deep hp config')
        hp = {'eta': 0.1, 'seed':0, 'subsample': 0.75, 'colsample_bytree': 0.85,
        'gamma': 2.5, 'objective': 'binary:logistic', 'max_depth':8,
        'min_child_weight':0.4, 'silent':1}
    else:
        hp = {'eta': 0.1, 'seed':0, 'subsample': 0.8, 'colsample_bytree': 0.9,
        'gamma': 1.6, 'objective': 'binary:logistic', 'max_depth':6,
        'min_child_weight':1, 'silent':1}

    return hp


def balanced_sample(df, uspl=True, seed=42):
    features_l = [df[df['labels']==l].copy() for l in list(set(df['labels'].values))]
    lsz = [f.shape[0] for f in features_l]
    return pd.concat([f.sample(n = (min(lsz) if uspl else max(lsz)), replace = (not uspl)).copy() for f in features_l], axis=0 ).sample(frac=1, random_state=seed) 

def load_data(datasetName, ident, test_size=0.05, parquet=False, balance=False):
    
    """
    Read dataset as parquet or HDF5 format, splits into train, test sets
    """
    from sklearn.model_selection import train_test_split
    excludeFeatures = ['labels', 'mbc', 'deltae', 'nCands', 'evtNum', 'MCtype', 'channel']
    if parquet:
        dataset = pq.ParquetDataset(datasetName)
        pdf = dataset.read(nthreads=4).to_pandas()
        pdf = balanced_sample(pdf) if balance else pdf.sample(frac=1)
    else:
        pdf = pd.read_hdf(datasetName, key='df')

    # Split data into training, testing sets
    df_X_train, df_X_test, df_y_train, df_y_test = train_test_split(pdf.drop(excludeFeatures, axis = 1), pdf['labels'], test_size = test_size, random_state=42)

    dTrain = xgb.DMatrix(data = df_X_train, label = df_y_train, feature_names=df_X_train.columns.tolist())
    dTest = xgb.DMatrix(data = df_X_test, label = df_y_test, feature_names=df_X_train.columns.tolist())
    print('fNames: {}'.format(dTrain.feature_names))

    print('# Features: {} | # Train Samples: {} | # Test Samples: {}'.format(dTrain.num_col(),
     dTrain.num_row(), dTest.num_row()))

    # Save to XGBoost binary file for faster loading
    dTrain.save_binary(os.path.join('dmatrices', "dTrain_{}.buffer".format(ident)))
    dTest.save_binary(os.path.join('dmatrices', "dTest_{}.buffer".format(ident)))

    return dTrain, dTest


def train_hyp_config(data_train, data_test, hyp_params, num_boost_rounds, identifier):
    # Returns validation metric after training configuration for allocated resources
    # Inputs: data - DMatrix tuple: (train, test)

    # Add evaluation metrics for validation set
    hyp_params['eval_metric'] = 'error@0.5'
    pList = list(hyp_params.items())+[('eval_metric', 'auc')]

    # Number of boosted trees to construct
    nTrees = num_boost_rounds
    # Specify validation set to watch performance
    dTrain, dTest = data_train, data_test
    evalList  = [(dTrain,'train'), (dTest,'eval')]

    print("Starting model training\n")
    start_time = time.time()
    # Train the model using the above parameters
    bst = xgb.train(params = pList, dtrain = dTrain, evals = evalList, num_boost_round = nTrees,
                    early_stopping_rounds = 256, verbose_eval = 25)

    delta_t = time.time() - start_time
    print("Training ended. Elapsed time: (%.3f s)." %(delta_t))
    pprint(bst.attributes())

    evalDict = {'auc': float(bst.attr('best_score')), 'error@0.5': bst.attr('best_msg').split('\t')[-2],
                'best_iteration': int(bst.attr('best_iteration'))}

    model_output = os.path.join('models', '{}.model'.format(identifier))
    bst.save_model(model_output)

    return bst, evalDict


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

def save_results(results):
    # Save dictionary of results to json
    timestamp = time.strftime("%b_%d_%H:%M")
    output = os.path.join('models', args.id + timestamp + '.json')
    with open(output, 'w') as f:
        json.dump(results, f)
    print('Boosting complete. Model saved to {}'.format(output))

def plot_importances(bst):
    importances = bst.get_fscore()
    df_importance = pd.DataFrame({'Importance': list(importances.values()), 'Feature': list(importances.keys())})
    df_importance.sort_values(by = 'Importance', inplace = True)
    df_importance[-20:].plot(kind = 'barh', x = 'Feature', color = 'orange', figsize = (15,15),
                                     title = 'Feature Importances')
    plt.savefig(os.path.join('graphs', args.id + 'xgb_importances.pdf'), format='pdf', dpi=1000)

def plot_ROC_curve(y_true, y_pred, meta = ''):
    from sklearn.metrics import roc_curve, auc

    # Compute ROC curve, integrate
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.axes([.1,.1,.8,.7])
    plt.figtext(.5,.9, r'$\mathrm{Receiver \;Operating \;Characteristic}$', fontsize=15, ha='center')
    plt.figtext(.5,.85, meta, fontsize=10,ha='center')
    plt.plot(fpr, tpr, color='darkorange',
                     lw=2, label='ROC (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=1.0, linestyle='--')
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel(r'$\mathrm{False \;Positive \;Rate}$')
    plt.ylabel(r'$\mathrm{True \;Positive \;Rate}$')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join('graphs', 'pi0veto_ROC.pdf'), format='pdf', dpi=1000)
    plt.gcf().clear()

def diagnostics(dTest, bst, hp, identifier):
    xgb_pred = bst.predict(dTest)
    y_pred = np.greater(xgb_pred, 0.5)
    y_true = dTest.get_label()
    #y_true = df_y_test.values

    test_accuracy = np.equal(y_pred, y_true).mean()
    print('Test accuracy: {}'.format(test_accuracy))

    plot_ROC_curve(y_true = y_true, y_pred = xgb_pred,
            meta = r'xgb: {} - $\eta$: {}, depth: {}'.format(identifier, hp['eta'], hp['max_depth']))
    #plot_importances(bst)
    print('Diagnostic graphs saved to graphs/')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data_file', help = 'Path to training dataset')
    parser.add_argument('id', help = 'Decay identifier')
    parser.add_argument('-n', '--num_boost_rounds', type = int,  help = 'Number of boosting rounds')
    parser.add_argument('-r', '--randomhp', help = 'Use random hyperparameters', action = 'store_true')
    parser.add_argument('-deep', '--deeptrees', help = 'Deeper tree config', action = 'store_true')
    parser.add_argument('-diag', '--diagnostics', help = 'Save diagnostics to file', action = 'store_true')
    parser.add_argument('-bal', '--balanced', help = 'Use scale_pos_weight over dataset', action = 'store_true')
    args = parser.parse_args()

    print('Loading dataset from: %s with test size 0.05' %(args.data_file))
    dTrain, dTest = load_data(args.data_file, args.id, parquet=False)

    # Get hyperparameter config
    if args.randomhp:
        print('Using random hp config')
        hp = get_hyp_config()
    else:
        print('Using default hp config')
        hp = hp_default_config(args.deeptrees)

    if args.balanced:
        num_T = dTrain.num_row()*dTrain.get_label().mean()
        hp['scale_pos_weight'] = (dTrain.num_row()-num_T)/num_T
        print('Balancing classes: F/T ratio: {}'.format(hp['scale_pos_weight']))
    num_boost_rounds=612
    if args.num_boost_rounds:
        num_boost_rounds=args.num_boost_rounds

    # Start boosting
    t0 = time.time()
    print('Boosting for {} iterations'.format(num_boost_rounds))
    bst, results = train_hyp_config(dTrain, dTest, hyp_params = hp, num_boost_rounds = num_boost_rounds, identifier = args.id)
    save_results(results)

    # Generate diagnostic summary
    if args.diagnostics:
        diagnostics(dTest, bst, hp, args.id)

