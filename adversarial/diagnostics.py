# -*- coding: utf-8 -*-
# Diagnostic helper functions for Tensorflow session
import tensorflow as tf
import os, time
from tensorflow.python.client import device_lib

def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    #return local_device_protos
    print('Available GPUs:')
    print([x.name for x in local_device_protos if x.device_type == 'GPU'])

def plot_ROC_curve(network_output, y_true, identifier, meta = ''):
    from sklearn.metrics import roc_curve, auc
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Compute ROC curve, integrate
    y_pred = network_output
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)

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
    plt.savefig(os.path.join('graphs', 'val_{}'.format(identifier) + 'ROC.pdf'), format='pdf', dpi=1000)
    plt.show()
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
    plt.savefig(os.path.join('graphs', 'val_{}'.format(identifier) + 'SEvBR.pdf'), format='pdf', dpi=1000)
    plt.show()
    plt.gcf().clear()

def plot_distributions(model, epoch, sess, handle, nbins=64, notebook=True):
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import seaborn as sns
    plt.style.use('seaborn-darkgrid')
    plt.style.use('seaborn-talk')
    feed_dict = {model.training_phase: False, model.handle: handle}
    ancillary, label, p, transformed_p = sess.run([model.ancillary, model.label, model.p, model.transform], feed_dict=feed_dict)
    assert ancillary.shape[0] == p.shape[0], 'Dimension mismatch along axis 0!'
    pdf=pd.DataFrame({'deltae': ancillary[:,0], 'mbc': ancillary[:,1]})
    pdf = pdf.assign(labels=label, preds=p[:,1], logit_p=transformed_p)

    def normPlot(variable, pdf, epoch, signal, nbins=50, bkg_rejection=0.995):
        titles={'mbc': r'$M_{bc}$ (GeV)', 'deltae': r'$\Delta E$ (GeV)', 'daughterInvM': r'$M_{X_q}$ (GeV)'}
        bkg = pdf[pdf['labels']<0.5]
        post_bkg = bkg.nlargest(int(bkg.shape[0]*(1-bkg_rejection)), columns=['preds'])
        threshold = post_bkg['preds'].min()
        print('Post-selection:', post_bkg.shape[0])
        sns.distplot(post_bkg[variable], hist=True, kde=True, label='Background - {} rejection'.format(bkg_rejection), bins=nbins)
        sns.distplot(bkg[variable], hist=True, kde=True, label='Background', bins=nbins)
        if signal:
            sig = pdf[pdf['labels']==1]
            post_sig = pdf[(pdf['labels']==1) & (pdf['preds']>threshold)]
            sns.distplot(post_sig[variable], hist=True, kde=True, label='Signal post-cut', bins=nbins)
            sns.distplot(sig[variable], hist=True, kde=True, label='Signal', bins=nbins)
        plt.xlabel(r'{}'.format(titles[variable]))
        plt.ylabel(r'Normalized events/bin')
        plt.legend(loc = "best")
        if notebook:
            plt.show()
            plt.savefig('graphs/{}_adv-ep{}-nb.pdf'.format(variable, epoch), bbox_inches='tight',format='pdf', dpi=1000)
        plt.gcf().clear()

    def binscatter(variable, x, y, nbins=69):
        titles={'mbc': r'$M_{bc}$ (GeV)', 'deltae': r'$\Delta E$ (GeV)', 'daughterInvM': r'$M_{X_q}$ (GeV)'}
        n,_ = np.histogram(x, nbins)
        sy, _ = np.histogram(x, bins=nbins, weights=y)
        sy2, _ = np.histogram(x, bins=nbins, weights=y*y)
        mean = sy / n
        std = np.sqrt(np.square(sy2/n - mean*mean))/4
        bins = (_[1:] + _[:-1])/2
        plt.errorbar(bins, mean, yerr=std, fmt='ro', label='Traditional NN', markersize=6, alpha=0.8)
        plt.xlabel(r'{}'.format(titles[variable]))
        plt.ylabel('NN Posterior')
        plt.legend(loc='best')
        # sns.regplot(bins,mean,order=1, marker='.',color='r')
        if notebook:
            plt.show()
            # plt.savefig('graphs/nn-{}_adv-ep{}-nb.pdf'.format(variable, epoch), bbox_inches='tight',format='pdf', dpi=1000)
        plt.gcf().clear()

    normPlot('mbc', pdf, epoch=epoch, signal=False)
    normPlot('deltae', pdf, epoch=epoch, signal=False)
    df_sig = pdf[pdf['labels']==1]
    df_bkg = pdf[pdf['labels']<0.5]
    binscatter('mbc', df_bkg['mbc'], df_bkg['preds'])
    binscatter('deltae', df_bkg['deltae'], df_bkg['preds'])


def run_diagnostics(model, config, directories, sess, saver, train_handle,
        test_handle, global_step, nTrainExamples, start_time, v_auc_best, epoch=None):
    t0 = time.time()
    if not epoch:
        epoch = int(global_step*config.batch_size/nTrainExamples)
    step = global_step*config.batch_size-epoch*nTrainExamples

    improved = ''
    sess.run(tf.local_variables_initializer())
    feed_dict_train = {model.training_phase: False, model.handle: train_handle}
    feed_dict_test = {model.training_phase: False, model.handle: test_handle}
    t_acc, t_loss, t_auc, t_summary = sess.run([model.accuracy, model.cross_entropy, model.auc_op, model.merge_op],
                                        feed_dict = feed_dict_train)
    v_acc, v_loss, v_auc, v_summary = sess.run([model.accuracy, model.cross_entropy, model.auc_op, model.merge_op],
                                        feed_dict = feed_dict_test)
    model.train_writer.add_summary(t_summary, global_step)
    model.test_writer.add_summary(v_summary, global_step)

    if v_auc > v_auc_best:
        v_auc_best = v_auc
        improved = '[*]'
        if epoch>5:
            save_path = saver.save(sess,
                        os.path.join(directories.checkpoints, 'vDNN_{}_{}_epoch{}.ckpt'.format(config.mode, config.channel, epoch)),
                        global_step=epoch)
            print('Graph saved to file: {}'.format(save_path))
    if epoch % 10 == 0:
        save_path = saver.save(sess, os.path.join(directories.checkpoints, 'vDNN_{}_{}_epoch{}.ckpt'.format(config.mode, 
            config.channel, epoch)), global_step=epoch) 
        print('Graph saved to file: {}'.format(save_path))
    print('Epoch {}, Step {} | Training Acc: {:.3f} | Test Acc: {:.3f} | Test Loss: {:.3f} | Test AUC: {:.3f} | Rate: {} examples/s ({:.2f} s) {}'.format(epoch, step, t_acc, v_acc, v_loss, v_auc, int(config.batch_size/(time.time()-t0)), time.time() - start_time, improved))

    return epoch, v_auc_best


def run_adv_diagnostics(model, config, directories, sess, saver, train_handle,
        test_handle, global_step, nTrainExamples, start_time, v_auc_best, epoch=None):
    t0 = time.time()
    if not epoch:
        epoch = int(global_step*config.batch_size/nTrainExamples)
    step = global_step*config.batch_size-epoch*nTrainExamples

    improved = ''
    sess.run(tf.local_variables_initializer())
    feed_dict_train = {model.training_phase: False, model.handle: train_handle}
    feed_dict_test = {model.training_phase: False, model.handle: test_handle}
    t_acc, t_loss, t_auc, t_summary = sess.run([model.accuracy, model.cross_entropy, model.auc_op, model.merge_op],
                                        feed_dict = feed_dict_train)
    v_ops = [model.accuracy, model.cross_entropy, model.auc_op, model.total_loss, model.merge_op]
    v_acc, v_loss, v_auc, v_total, v_summary = sess.run(v_ops, feed_dict=feed_dict_test)
    model.train_writer.add_summary(t_summary, global_step)
    model.test_writer.add_summary(v_summary, global_step)

    if v_auc > v_auc_best:
        v_auc_best = v_auc
        improved = '[*]'
        if epoch>5:
            save_path = saver.save(sess,
                        os.path.join(directories.best_checkpoints, 'vDNN_{}_{}_epoch{}.ckpt'.format(config.mode, config.channel, epoch)),
                        global_step = epoch)
            print('Graph saved to file: {}'.format(save_path))

    if epoch % 16 == 0:
        save_path = saver.save(sess, os.path.join(directories.checkpoints, 'vDNN_{}_{}_epoch{}.ckpt'.format(config.mode, 
            config.channel, epoch)), global_step=epoch) 
        print('Graph saved to file: {}'.format(save_path))

    print('Epoch {}, Step {} | Training Acc: {:.3f} | Test Acc: {:.3f} | Test Loss: {:.3f} | Test AUC: {:.3f} | Total loss: {:.3f} | Rate: {} examples/s ({:.2f} s) {}'.format(epoch, step, t_acc, v_acc, v_loss, v_auc, v_total, int(config.batch_size/(time.time()-t0)), time.time() - start_time, improved))

    return epoch, v_auc_best

