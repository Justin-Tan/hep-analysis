#!/usr/bin/env python
# Quick script to make predictions with saved model

import numpy as np
import pandas as pd
import xgboost as xgb
import sys, time, os, json
import argparse

def plot_ROC_curve(y_true, y_pred, meta = ''):
    from sklearn.metrics import roc_curve, auc

    # Compute ROC curve, integrate
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)

    print('AUC: {}'.format(roc_auc))
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
    plt.savefig(os.path.join('graphs', 'val_{}_{}'.format(args.channel, args.mode) + 'ROC.pdf'), format='pdf', dpi=1000)
    plt.gcf().clear()

def plot_importances(bst):
    importances = bst.get_fscore()
    df_importance = pd.DataFrame({'Importance': list(importances.values()), 'Feature': list(importances.keys())})
    df_importance.sort_values(by = 'Importance', inplace = True)
    df_importance[-20:].plot(kind = 'barh', x = 'Feature', color = 'orange', figsize = (15,15),
                                     title = 'Feature Importances')
    plt.savefig(os.path.join('graphs', 'val_{}_{}'.format(args.channel, args.mode) + 'xgb_rankings.pdf'), format='pdf', dpi=1000)

def load_test_data(fname, mode, channel):
    df = pd.read_hdf(fname, 'df')
    # Split data into training, testing sets
    df_X_test = df.drop(['labels', 'mbc', 'deltae'], axis = 1)
    df_y_test = df['labels']
    dTest = xgb.DMatrix(data = df_X_test.values, label = df_y_test.values, feature_names = df_X_test.columns)

    # Save to XGBoost binary file for faster loading
    binary_path = os.path.join('dmatrices', "dVal" + mode + channel + ".buffer")
    dTest.save_binary(binary_path)

    return dTest, binary_path

def diagnostics(dTest, df_y_test, bst):
    xgb_pred = bst.predict(dTest)
    y_pred = np.greater(xgb_pred, 0.5)
    y_true = df_y_test.values

    test_accuracy = np.equal(y_pred, y_true).mean()
    print('Test accuracy: {}'.format(test_accuracy))

    plot_ROC_curve(y_true = df_y_test.values, y_pred = xgb_pred,
            meta = 'xgb: {} - {} | eta: {}, depth: {}'.format(args.channel, args.mode, hp['eta'], hp['max_depth']))
    plot_importances(bst)
    print('Diagnostic graphs saved to graphs/')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data_file', help = 'Path to dataset in HDF5 format')
    parser.add_argument('model', help = 'Path to saved XGB model')
    parser.add_argument('channel', help = 'Decay channel')
    parser.add_argument('mode', help = 'Background type')
    args = parser.parse_args()

    # Load saved data
    print('Loading validation set from: {}'.format(args.data_file))
    dVal, binary_path = load_test_data(args.data_file, args.mode, args.channel)
    valDMatrix = xgb.Dmatrix(binary_path)

    # Load saved model, make predictions
    print('Using saved model {}'.format(args.model))
    bst = xgb.Booster({'nthread':16})
    bst.load_model(args.xgb_model)
    diagnostics(dataDMatrix, df_y_test, bst)
