#!/usr/bin/env python
# Quick script to make predictions with saved model

import numpy as np
import pandas as pd
import xgboost as xgb
import sys, time, os, json
import argparse
import matplotlib.pyplot as plt
import seaborn as sns

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

def load_test_data(val_fname, mode, channel):
    df = pd.read_hdf(val_fname, 'df')
    # Split data into training, testing sets
    df_X_test = df.drop(['labels', 'mbc', 'deltae'], axis = 1)
    df_y_test = df['labels']
    dTest = xgb.DMatrix(data = df_X_test.values, label = df_y_test.values.astype(int),
    feature_names = df_X_test.columns)

    # Save to XGBoost binary file for faster loading
    binary_path = os.path.join('dmatrices', "dVal" + mode + channel + ".buffer")
    dTest.save_binary(binary_path)

    return df, dTest, binary_path

def truncate_tails(hist, nsigma = 5):
    # Removes feature outliers above nsigma stddevs away from mean
    hist = hist[hist > np.mean(hist)-nsigma*np.std(hist)]
    hist = hist[hist < np.mean(hist)+nsigma*np.std(hist)]
    return hist

def plot_fit_variables(df_sig, nbins = 50, columns=None):
    # Accepts events classified as signal, and plots key fit distributions
    # and correlations between them
    sea_green = '#54ff9f'
    steel_blue = '#4e6bbd'

    for variable in columns:
        d_sig = truncate_tails(data_sig[variable].values,5)
        d_bkg = truncate_tails(data_bkg[variable].values,5)

        sns.distplot(d_sig,color = sea_green, hist=True, kde = False, norm_hist = True, label = r'$\mathrm{Signal}$',bins=nbins)
        sns.distplot(d_bkg,color = steel_blue, hist=True,label = r'$\mathrm{Background}$',kde=False,norm_hist=True,bins=nbins)

#         sns.kdeplot(data_array_bkg, color = steel_blue, label = 'Background',shade =True)
#         sns.kdeplot(d_cfd, color = crimson_tide, label = 'Crossfeed',shade =True)
#         sns.kdeplot(d_gen, color = yellow, label = 'Generic',shade =True)

        plt.title(variable)
        plt.autoscale(enable=True, axis='x', tight=False)
        plt.xlabel(variable)
        plt.ylabel(r'$\mathrm{Normalized \; events/bin}$')
        plt.legend(loc = "best")
        #plt.title(r"$\mathrm{"+variable+"{\; - \; (B \rightarrow K^+ \pi^0) \gamma$")
        #plt.xlim(-1.0,0.98)
        #plt.ylim(0,3.3)
        #plt.xlabel(r'$|(p_B)_{CMS}| \; [GeV/c]$')
        plt.savefig('graphs/' + mode + variable + '.pdf', bbox_inches='tight',format='pdf', dpi=1000)
        plt.show()
        plt.gcf().clear()

def diagnostics(dTest, bst):
    xgb_pred = bst.predict(dTest)
    y_pred = np.greater(xgb_pred, 0.5)
    y_true = dTest.get_label()
    #y_true = df_y_test.values

    test_accuracy = np.equal(y_pred, y_true).mean()
    print('Test accuracy: {}'.format(test_accuracy))

    plot_ROC_curve(y_true = y_true, y_pred = xgb_pred,
            meta = 'xgb: {} - {} | eta: {}, depth: {}'.format(args.channel, args.mode, 0.1, 6))#, hp['eta'], hp['max_depth']))
    plot_importances(bst)
    print('Diagnostic graphs saved to graphs/')

    return y_pred

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data_file', help = 'Path to dataset in HDF5 format')
    parser.add_argument('model', help = 'Path to saved XGB model')
    parser.add_argument('channel', help = 'Decay channel')
    parser.add_argument('mode', help = 'Background type')
    args = parser.parse_args()

    # Load saved data
    print('Loading validation set from: {}'.format(args.data_file))
    df_val, dVal, binary_path = load_test_data(args.data_file, args.mode, args.channel)
    valDMatrix = xgb.DMatrix(binary_path)

    # Load saved model, make predictions
    print('Using saved model {}'.format(args.model))
    bst = xgb.Booster({'nthread':16})
    bst.load_model(args.model)
    diagnostics(valDMatrix, bst)
