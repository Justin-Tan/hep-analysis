import numpy as np
import pandas as pd
import sys, time, os, json
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
plt.ioff()
import seaborn as sns
import xgboost as xgb
import argparse

def plot_ROC_curve(y_true, y_pred, meta = ''):
    from sklearn.metrics import roc_curve, auc

    # Compute ROC curve, integrate
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)

    print('AUC: {}'.format(roc_auc))
    plt.figure()
    plt.axes([.1,.1,.8,.7])
    plt.figtext(.5,.9, r'Receiver Operating Characteristic', fontsize=15, ha='center')
    plt.figtext(.5,.85, meta, fontsize=10,ha='center')
    plt.plot(fpr, tpr, color='darkorange',
                     lw=2, label='ROC (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=1.0, linestyle='--')
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel(r'False Positive Rate')
    plt.ylabel(r'True Positive Rate')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join('graphs', 'val_{}_{}'.format(args.channel, args.mode) + 'ROC.pdf'), format='pdf', dpi=1000)
    plt.show()
    plt.gcf().clear()

def load_test_data(fname, mode, channel):
    df = pd.read_hdf(fname, 'df')
    # Split data into training, testing sets
    df_X_test = df.drop(['labels', 'mbc', 'deltae'], axis = 1)
    df_y_test = df['labels']
    dTest = xgb.DMatrix(data = df_X_test.values, label = df_y_test.values.astype(int),
    feature_names = df_X_test.columns)

    # Save to XGBoost binary file for faster loading
    binary_path = os.path.join("dVal" + mode + channel + ".buffer")
    dTest.save_binary(binary_path)

    return df, dTest, binary_path

def truncate_tails(hist, nsigma = 5):
    # Removes feature outliers above nsigma stddevs away from mean
    hist = hist[hist > np.mean(hist)-nsigma*np.std(hist)]
    hist = hist[hist < np.mean(hist)+nsigma*np.std(hist)]
    return hist

def plot_fit_variables(df_fit, nbins = 42, kde = True):
    # Accepts events classified as signal, and plots key fit distributions
    # and correlations between them
    sea_green = '#54ff9f'
    steel_blue = '#6495ED'
    orange = '#ffa500'
    color_dict = {'mbc': sea_green, 'deltae': steel_blue, 'logit-scores': orange}

    for variable in df_fit.columns:
        d_sig = truncate_tails(df_fit[variable].values,4)
        sns.distplot(d_sig,color = color_dict[variable], hist=True, kde = kde, norm_hist = False, label = r'$\mathrm{Signal}$',bins=nbins,
                     hist_kws=dict(edgecolor="0.5", linewidth=1))

        plt.title(variable)
        plt.xlabel(variable)
        plt.ylabel('Events/bin')
        plt.legend(loc = "best")
        plt.show()
        plt.gcf().clear()
        #plt.title(r"$\mathrm{"+variable+"{\; - \; (B \rightarrow K^+ \pi^0) \gamma$")
        #plt.xlim(-1.0,0.98)
        #plt.ylim(0,3.3)
        #plt.xlabel(r'$|(p_B)_{CMS}| \; [GeV/c]$')
        plt.savefig(os.path.join('graphs', args.channel + args.mode + variable + '.pdf'),
                bbox_inches='tight',format='pdf', dpi=1000)

def diagnostics(dTest, bst):
    xgb_pred = bst.predict(dTest)
    y_pred = np.greater(xgb_pred, 0.5)
    y_true = dTest.get_label()
    #y_true = df_y_test.values

    test_accuracy = np.equal(y_pred, y_true).mean()
    print('Test accuracy: {}'.format(test_accuracy))

    plot_ROC_curve(y_true = y_true, y_pred = xgb_pred,
            meta = 'xgb: {} - {} | eta: {}, depth: {}'.format(args.channel, args.mode, 0.1, 6))#, hp['eta'], hp['max_depth']))
    print('Diagnostic graphs saved to graphs/')

    return y_pred, xgb_pred

def plot_pairgrid(df_fit):
    def hexbin(x, y, color, max_series=None, min_series=None, **kwargs):
        # cmap = sns.light_palette(color, as_cmap=True)
        cmap = 'BuGn'
        ax = plt.gca()
        xmin, xmax = min_series[x.name], max_series[x.name]
        ymin, ymax = min_series[y.name], max_series[y.name]
        plt.hexbin(x, y, gridsize=60, cmap=cmap, extent=[xmin, xmax, ymin, ymax], **kwargs)

    steel_blue = '#6495ED'
    g = sns.PairGrid(df_fit, size = 5)
    g.map_diag(sns.distplot, bins = 40, kde = False, hist_kws=dict(edgecolor="0.5", linewidth=1))
    # g.map_diag(plt.hist, alpha = 0.5, bins = 40)
    g.map_lower(sns.kdeplot, cmap = "Blues", shade = True, shade_lowest = False, n_levels = 30)
    g.map_upper(hexbin, min_series = df_fit.min(), max_series = df_fit.max(), alpha=0.5)#, cmap= 'cubehelix')
    plt.savefig(os.path.join('graphs', 'val_{}_{}'.format(args.channel, args.mode) + 'pairgrid.pdf'), format='pdf', dpi=1000)
    plt.gcf().clear()

def classic_hexplot(x, y, x_title, y_title, xlim = None, ylim = None):
    plt.hexbin(x, y, cmap = 'jet', gridsize = 150)
    plt.colorbar()
    plt.xlim(5.26,5.289)
    plt.ylim(-0.25,0.25)
    plt.xlabel("${}$".format(x_title))
    plt.ylabel("${}$".format(y_title))
    plt.show()

def kdeplot2D(x, y, x_title, y_title, xlim = None, ylim = None):
    sns.set(style="ticks")
    cmap = sns.cubehelix_palette(as_cmap=True, dark=0, light=1, reverse = False, rot=-.4)
    kdeplot = sns.jointplot(x, y, kind="kde", size = 12, n_levels = 70, space=0, xlim = xlim, ylim = ylim, gridsize=200, color="#4CB391", cmap = cmap)
    kdeplot.set_axis_labels('${}$'.format(x_title), '${}$'.format(y_title))
    plt.savefig(os.path.join('graphs', 'val_{}_{}_{}-{}'.format(args.channel, args.mode, x.name, y.name) + 'kde.pdf'), format='pdf', dpi=1000)
    plt.show()

def hexplot2D(x, y, x_title, y_title, xlim = None, ylim = None):
    sns.set(style="ticks")
    cmap = sns.cubehelix_palette(as_cmap=True, dark=0, light=1, reverse = False, rot=-.4)
    hexplot = sns.jointplot(x, y, kind="hex", size = 15, space=0, gridsize=50, color="#4CB391", cmap = cmap, extent=[xlim[0], xlim[1], ylim[0], ylim[1]],
                           xlim = xlim, ylim = ylim)
                          #  extent=[np.min(x.values), np.max(x.values), np.min(x.values), np.max(x.values)])
    sns.plt.subplots_adjust(left=0.2, right=0.8, top=0.8, bottom=0.2)  # shrink fig so cbar is visible
    cax = hexplot.fig.add_axes([.85, .25, .05, .4])  # x, y, width, height
    sns.plt.colorbar(cax=cax)
    hexplot.set_axis_labels('${}$'.format(x_title), '${}$'.format(y_title))
    plt.savefig(os.path.join('graphs', 'val_{}_{}_{}-{}'.format(args.channel, args.mode, x.name, y.name) + 'hex.pdf'), format='pdf', dpi=1000)
    plt.show()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('data_file', help = 'Path to dataset in HDF5 format')
    parser.add_argument('model', help = 'Path to saved XGB model')
    parser.add_argument('channel', help = 'Decay channel')
    parser.add_argument('mode', help = 'Background type')
    args = parser.parse_args()

    print('Loading validation set from: {}'.format(args.data_file))
    df_val, dVal, _ = load_test_data(args.data_file, args.channel, args.mode)

    print('Using saved model {}'.format(args.model))
    bst = xgb.Booster({'nthread':16})
    bst.load_model(args.model)
    y_pred, xgb_pred = diagnostics(dVal, bst)
    df_sig = df_val.loc[y_pred]

    from scipy.special import logit
    logit_scores = logit(xgb_pred[y_pred])

    df_fit = pd.DataFrame({'mbc': df_sig['mbc'], 'deltae': df_sig['deltae'], 'logit-scores': logit_scores})
    limits = {'mbc': (5.265, 5.29), 'deltae': (-0.25, 0.25), 'scores': (0,6)}

    plot_pairgrid(df_fit)
    plot_fit_variables(df_fit)
    hexplot2D(df_sig['mbc'], df_sig['deltae'], 'Mbc', '\Delta E', xlim = limits['mbc'], ylim = limits['deltae'])
    hexplot2D(df_fit['mbc'], df_fit['logit-scores'], 'Mbc', 'Logit(scores)', xlim = limits['mbc'], ylim = limits['scores'])
    hexplot2D(df_fit['deltae'], df_fit['logit-scores'], 'Mbc', 'Logit(scores)', xlim = limits['deltae'], ylim = limits['scores'])
