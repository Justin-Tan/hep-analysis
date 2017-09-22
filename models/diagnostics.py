# -*- coding: utf-8 -*-
# Tensorflow Implementation of the Scaled ELU function and Dropout
import tensorflow as tf
import os, time

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


def run_diagnostics(model, config, directories, sess, saver, train_handle,
            test_handle, global_step, nTrainExamples, start_time, v_auc_best, n_checkpoints):
    t0 = time.time()
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
        if not n_checkpoints%5 and epoch>2:
            save_path = saver.save(sess,
                        os.path.join(directories.checkpoints, 'vDNN_{}_{}_epoch{}.ckpt'.format(config.mode, config.channel, epoch)),
                        global_step = epoch)
            print('Graph saved to file: {}'.format(save_path))
            n_checkpoints+=1

    print('Epoch {}, Step {} | Training Acc: {:.3f} | Test Acc: {:.3f} | Test Loss: {:.3f} | Test AUC: {:.3f} | Rate: {} examples/s ({:.2f} s) {}'.format(epoch, step, t_acc, v_acc, v_loss, v_auc, int(config.batch_size/(time.time()-t0)), time.time() - start_time, improved))

    return epoch, v_auc_best
