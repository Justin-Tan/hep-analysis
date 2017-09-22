#!/usr/bin/env python
# Quick script to make predictions with saved model

import numpy as np
import pandas as pd
import xgboost as xgb
import sys, time, os, json
import argparse

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.special import logit


def plot_ROC_curve(y_true, y_pred, meta = ''):
    from sklearn.metrics import roc_curve, auc

    # Compute ROC curve, integrate
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)

    print('Plotting AUC')
    print('AUC: {}'.format(roc_auc))
    plt.figure()
    plt.axes([.1,.1,.8,.75])
    plt.figtext(.5,.95, r'Receiver Operating Characteristic', fontsize=15, ha='center')
    plt.figtext(.5,.9, meta, fontsize=10,ha='center')
    plt.plot(fpr, tpr, color='darkorange',
                     lw=2, label='ROC (area = %0.3f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=1.0, linestyle='--')
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel(r'False Positive Rate')
    plt.ylabel(r'True Positive Rate')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join('graphs', 'val_{}'.format(args.id) + 'ROC.pdf'), format='pdf', dpi=1000)
    plt.gcf().clear()

    print('Plotting signal efficiency versus background rejection')
    plt.figure()
    plt.axes([.1,.1,.8,.75])
    plt.figtext(.5,.95, r'Signal Efficiency v. Background Rejection', fontsize=15, ha='center')
    plt.figtext(.5,.9, meta, fontsize=10,ha='center')
    plt.plot(tpr, 1-fpr, color='springgreen',
                     lw=2, label='ROC (area = %0.3f)' % roc_auc)
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel(r'$\epsilon_S$')
    plt.ylabel(r'$1-\epsilon_B$')
    plt.legend(loc="upper right")
    plt.savefig(os.path.join('graphs', 'val_{}'.format(args.id) + 'SEvBR.pdf'), format='pdf', dpi=1000)
    plt.gcf().clear()

def load_data(df_fname, identifier):

    df = pd.read_hdf(df_fname, 'df', start=0, stop=25000)
    df_X = df.drop(['labels', 'mbc', 'deltae'], axis = 1)
    df_y = df['labels']
    dMatrix = xgb.DMatrix(data = df_X.values, label = df_y.values.astype(int),
    feature_names = df_X.columns)
    df_fit = df[['mbc', 'deltae']]
    del df, df_X

    return dMatrix, df_fit, df_y

def truncate_tails(hist, nsigma = 5):
    # Removes feature outliers above nsigma stddevs away from mean
    hist = hist[hist > np.mean(hist)-nsigma*np.std(hist)]
    hist = hist[hist < np.mean(hist)+nsigma*np.std(hist)]
    return hist

def plot_fit_variables(df_sig, df_bkg, identifier, nbins = 50, columns=['mbc', 'deltae']):
    # Accepts events classified as signal, and plots key fit distributions
    # and correlations between them
    sea_green = '#54ff9f'
    steel_blue = '#4e6bbd'
    latex_dict = {'mbc': r'$M_{bc}$', 'deltae': r'$\Delta E$'}

    for variable in columns:
        d_sig = truncate_tails(df_sig[variable].values,4)
        d_bkg = truncate_tails(df_bkg[variable].values,4)

        sns.distplot(d_sig,color = sea_green, hist=True, kde = False, norm_hist = False, label = r'Signal (True $\pi^0$)',bins=nbins,
                        hist_kws=dict(edgecolor="0.5", linewidth=1))
        sns.distplot(d_bkg,color = steel_blue, hist=True,label = r'Background (Non $\pi^0$)',kde=False,norm_hist=False,bins=nbins,
                        hist_kws=dict(edgecolor="0.5", linewidth=1))
        # sns.kdeplot(data_array_bkg, color = steel_blue, label = 'Background',shade =True)

        plt.title(latex_dict[variable])
        plt.autoscale(enable=True, axis='x', tight=False)
        plt.xlabel(latex_dict[variable])
        plt.ylabel(r'Normalized Events/bin')
        plt.legend(loc = "best")
        #plt.xlim(-1.0,0.98)
        #plt.ylim(0,3.3)
        #plt.xlabel(r'$|(p_B)_{CMS}| \; [GeV/c]$')
        plt.savefig('graphs/{}_{}.pdf'.format(identifier, variable), bbox_inches='tight',format='pdf', dpi=1000)
        plt.gcf().clear()

def plot_importances(bst):
    print('Plotting variable importances')
    importances = bst.get_fscore()
    df_importance = pd.DataFrame({'Importance': list(importances.values()), 'Feature': list(importances.keys())})
    df_importance.sort_values(by = 'Importance', inplace = True)
    df_importance[-20:].plot(kind = 'barh', x = 'Feature', color = 'orange', figsize = (15,15),
                                     title = 'Feature Importances')
    plt.savefig(os.path.join('graphs', 'val_{}'.format(args.id) + 'xgb_rankings.pdf'), format='pdf', dpi=1000)

def normalize_weights(x):
    # Weights to normalize output histograms
    normalizing_weights = np.ones(x.shape[0])*1/x.shape[0]
    return normalizing_weights

def TT_check(dTrain, dTest, identifier, meta = '', nbins = 50):
    # Plot neural network output for train, test instances to check overtraining
    print("Plotting overfitting check")
    sea_green = '#54ff9f'
    cornflower = '#6495ED'
    xgb_pred_train = bst.predict(dTrain)
    xgb_pred_test = bst.predict(dTest)
    y_train, y_test = dTrain.get_label(), dTest.get_label()

    sig_indices_train, bkg_indices_train = np.where(y_train == 1), np.where(y_train == 0)
    sig_indices_test, bkg_indices_test = np.where(y_test == 1), np.where(y_test == 0)
    sig_output_train, bkg_output_train = xgb_pred_train[sig_indices_train], xgb_pred_train[bkg_indices_train]
    sig_output_test, bkg_output_test = xgb_pred_test[sig_indices_test], xgb_pred_test[bkg_indices_test]

    plt.figure()
    plt.axes([.1,.1,.8,.8])
    plt.figtext(.5,.95, r'Probability that $\gamma$ originates from asymmetric $\pi^0$ decay', fontsize=12, ha='center')

    # Plot the training sample as filled histograms
    sns.distplot(sig_output_train, color = sea_green, label = r'Signal (True $\pi^0$)',bins = nbins, kde = False,
                 hist_kws=dict(weights = normalize_weights(sig_output_train), edgecolor="0.5", linewidth=1))
    sns.distplot(bkg_output_train, color = cornflower, label = r'Background (Non $\pi^0$)',bins=nbins, kde = False,
                 hist_kws=dict(weights = normalize_weights(bkg_output_train), edgecolor="0.5", linewidth=1))

    hist, bins = np.histogram(sig_output_test, bins = nbins, weights = normalize_weights(sig_output_test))
    center = (bins[:-1] + bins[1:])/2
    plt.errorbar(center, hist, fmt='.',c = sea_green, label = r'Signal (test)', markersize='10')
    hist, bins = np.histogram(bkg_output_test, bins = nbins, weights = normalize_weights(bkg_output_test))
    center = (bins[:-1] + bins[1:])/2
    plt.errorbar(center, hist, fmt='.',c = cornflower, label = r'Background (test)', markersize='10')

    plt.xlabel(r'$\pi^0$ probability')
    plt.ylabel(r'Normalized Entries/bin')
    plt.legend(loc='best')
    plt.savefig("graphs/{}_TTcheck_norm.pdf".format(identifier), format='pdf', dpi=1000)
    plt.gcf().clear()

    plt.figure()
    plt.axes([.1,.1,.8,.8])
    plt.figtext(.5,.95, r'Logit-transformed probability', fontsize=12, ha='center')

    # Plot the training sample as filled histograms
    sns.distplot(logit(sig_output_train), color = sea_green, label = r'Signal (True $\pi^0$)',bins = nbins, kde = False,
                 hist_kws=dict(weights = normalize_weights(sig_output_train), edgecolor="0.5", linewidth=1))
    sns.distplot(logit(bkg_output_train), color = cornflower, label = r'Background (Non $\pi^0$)',bins=nbins, kde = False,
                 hist_kws=dict(weights = normalize_weights(bkg_output_train), edgecolor="0.5", linewidth=1))

    hist, bins = np.histogram(logit(sig_output_test), bins = nbins, weights = normalize_weights(sig_output_test))
    center = (bins[:-1] + bins[1:])/2
    plt.errorbar(center, hist, fmt='.',c = sea_green, label = r'Signal (test)', markersize='10')
    hist, bins = np.histogram(logit(bkg_output_test), bins = nbins, weights = normalize_weights(bkg_output_test))
    center = (bins[:-1] + bins[1:])/2
    plt.errorbar(center, hist, fmt='.',c = cornflower, label = r'Background (test)', markersize='10')

    plt.xlabel(r'Logit-transformed $\pi^0$ probability')
    plt.ylabel(r'Normalized Entries/bin')
    plt.legend(loc='best')
    plt.savefig("graphs/{}_TTcheck_logitnorm.pdf".format(identifier), format='pdf', dpi=1000)
    plt.gcf().clear()

    plt.figure()
    plt.axes([.1,.1,.8,.8])
    plt.figtext(.5,.95, r'Probability that $\gamma$ originates from asymmetric $\pi^0$ decay', fontsize=12, ha='center')

    # Plot the training sample as filled histograms
    sns.distplot(sig_output_test, color = sea_green, label = r'Signal (True $\pi^0$)',bins = nbins, kde = False,
                hist_kws=dict(edgecolor="0.5", linewidth=1))
    sns.distplot(bkg_output_test, color = cornflower, label = r'Background (Non $\pi^0$)',bins=nbins, kde = False,
                hist_kws=dict(edgecolor="0.5", linewidth=1))

    plt.xlabel(r'$\pi^0$ probability')
    plt.ylabel(r'Entries/bin')
    plt.legend(loc='best')
    plt.savefig("graphs/{}_testscores.pdf".format(identifier), format='pdf', dpi=1000)
    plt.gcf().clear()

def diagnostics(dTrain, dTest, bst, identifier, cut = 0.5):
    xgb_pred = bst.predict(dTest)
    y_pred = np.greater(xgb_pred, cut)
    y_true = dTest.get_label()

    test_accuracy = np.equal(y_pred, y_true).mean()
    print('Test accuracy: {}'.format(test_accuracy))

    plot_ROC_curve(y_true = y_true, y_pred = xgb_pred,
            meta = 'xgb: {} | eta: {}, depth: {}'.format(identifier, 0.1, 6))#, hp['eta'], hp['max_depth']))

    TT_check(dTrain, dTest, identifier)
    return xgb_pred

def hexplot2D(x, y, x_title, y_title, xlim, ylim):
    sns.set(style="ticks")
    cmap = sns.cubehelix_palette(as_cmap=True, dark=0, light=1, reverse = False, rot=-.4)
    hexplot = sns.jointplot(x, y, kind="hex", size = 15, space=0, gridsize=50, color="#4CB391", cmap = cmap, extent=[xlim[0], xlim[1], ylim[0], ylim[1]], xlim = xlim, ylim = ylim)
                          #  extent=[np.min(x.values), np.max(x.values), np.min(x.values), np.max(x.values)])
    sns.plt.subplots_adjust(left=0.2, right=0.8, top=0.8, bottom=0.2)  # shrink fig so cbar is visible
    cax = hexplot.fig.add_axes([.85, .25, .05, .4])  # x, y, width, height
    sns.plt.colorbar(cax=cax)
    hexplot.set_axis_labels('${}$'.format(x_title), '${}$'.format(y_title))
    plt.savefig(os.path.join('graphs', 'val_{}_{}-{}'.format(args.id, x.name, 'logit_score') + 'hex.pdf'), format='pdf', dpi=1000)

def inspect_fitVs(df_fit, df_y, xgb_pred, identifier, cut = 0.5, nbins=50):
    logit_pred = logit(xgb_pred)
    print('Inspecting fit variables')

    sea_green = '#54ff9f'
    cornflower = '#6495ED'
    latex_dict = {'mbc': r'$M_{bc}$', 'deltae': r'$\Delta E$', 'gbt_out': r'Logit(score)'}
    y_pred = np.less(xgb_pred, cut)

    # Before classification
    pre_df_fit_sig = df_fit.iloc[df_y.values.astype(bool)]
    pre_df_fit_bkg = df_fit.iloc[np.logical_not(df_y.values.astype(bool))]
    plot_fit_variables(pre_df_fit_sig, pre_df_fit_bkg, 'pre-cut_pi0veto', nbins, columns=df_fit.columns)

    # After classification
    post_df_fit_sig = df_fit.iloc[np.logical_and(df_y.values, y_pred)]
    post_df_fit_bkg = df_fit.iloc[np.logical_and(np.logical_not(df_y.values), y_pred)]
    plot_fit_variables(post_df_fit_sig, post_df_fit_bkg, 'post-cut_pi0veto', nbins, columns=df_fit.columns)

    # Efficiency versus fit variable

    for variable in df_fit.columns:
        hexplot2D(df_fit[variable], logit_pred, latex_dict[variable], latex_dict['gbt_out'],
        xlim = (np.min(df_fit[variable]), np.max(df_fit[variable])), ylim = (np.min(logit_pred), np.max(logit_pred)))
        plt.figure()
        plt.axes([.1,.1,.8,.8])
        plt.figtext(.5,.95, r'Signal/Background Efficiency versus {}, Cut: {}'.format(latex_dict[variable], cut), fontsize=12, ha='center')

        sig_prehist, prebins = np.histogram(pre_df_fit_sig[variable], bins = nbins)
        sig_posthist, postbins = np.histogram(post_df_fit_sig[variable], bins = nbins)
        center = (postbins[:-1] + postbins[1:])/2
        sig_eff = sig_posthist/sig_prehist
        plt.errorbar(center, sig_eff, fmt='-', c = sea_green, label = r'S: True $\pi^0$', markersize='10')

        bkg_prehist, prebins = np.histogram(pre_df_fit_bkg[variable], bins = nbins)
        bkg_posthist, postbins = np.histogram(post_df_fit_bkg[variable], bins = nbins)
        center = (postbins[:-1] + postbins[1:])/2
        bkg_eff = bkg_posthist/bkg_prehist
        plt.errorbar(center, bkg_eff, fmt='-', c = cornflower, label = r'B: Non $\pi^0$', markersize='10')

        plt.xlabel(latex_dict[variable])
        plt.ylabel(r'Efficiency')
        plt.legend(loc='best')
        plt.savefig("graphs/{}_{}_SBeff.pdf".format(identifier, variable), format='pdf', dpi=1000)
        plt.gcf().clear()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('val_file', help = 'Path to validation dataset in HDF5 format')
    parser.add_argument('train_file', help = 'Path to training dataset in HDF5 format')
    parser.add_argument('model', help = 'Path to saved XGB model')
    parser.add_argument('id', help = 'Identifier')
    args = parser.parse_args()

    # Load saved data
    print('Loading validation set from: {}'.format(args.val_file))
    dVal, df_val_fit, df_val_y = load_data(args.val_file, args.id)
    print('Loading train set from: {}'.format(args.train_file))
    dTrain, df_train_fit, df_train_y = load_data(args.train_file, args.id)

    # plot_fit_variables(df_train_fit, df_val_fit)

    # Load saved model, make predictions
    print('Using saved model {}'.format(args.model))
    bst = xgb.Booster({'nthread':16})
    bst.load_model(args.model)

    xgb_pred = diagnostics(dTrain, dVal, bst, args.id)
    inspect_fitVs(df_val_fit, df_val_y, xgb_pred, 'pi0veto')
    print('Diagnostic graphs saved to graphs/')
