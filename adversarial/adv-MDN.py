#!/usr/bin/env python3
"""
Adversarial training to protect against distribution sculpting
Based on paper by Louppe et.al: arXiv:1611.01046
Justin Tan 2017
"""
import tensorflow as tf
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import glob, time, os
import argparse

import selu
from diagnostics import *

class config(object):
    # Network hyperparameters
    mode = 'adv-selu'
    channel = 'B2Xsy'
    keep_prob = 0.95
    num_epochs = 128
    batch_size = 256
    n_layers = 7
    adv_n_layers = 2
    adv_keep_prob = 1.0
    hidden_layer_nodes = [1024, 1024, 512, 512, 512, 256, 256]
    adv_hidden_nodes = [128,128]
    n_gaussians = 4
    ema_decay = 0.999
    learning_rate = 1e-5
    adv_learning_rate = 0.01
    adv_lambda = 7 
    cycles = 3 # Number of annealing cycles
    n_classes = 2
    epsilon = 1e-8
    builder = 'selu'
    K = 128
    adversary = True

class directories(object):
    train = '/data/projects/punim0011/jtan/spark/spark2tf/combined_b2sy_val.parquet'
    test = '/data/projects/punim0011/jtan/spark/spark2tf/combined_b2sy_test.parquet'
    val = '/data/projects/punim0011/jtan/spark/spark2tf/combined_b2sy_val.parquet'
    tensorboard = 'tensorboard'
    checkpoints = 'checkpoints'

def load_parquet(datasetName):
    from sklearn.model_selection import train_test_split
    excludeFeatures = ['labels', 'mbc', 'deltae', 'daughterInvM', 'nCands', 'evtNum', 'MCtype', 'channel']
    dataset = pq.ParquetDataset(datasetName)
    pdf = dataset.read(nthreads=8).to_pandas()
    pdf = pdf.sample(frac=1).reset_index(drop=True)
    features = pdf.drop(excludeFeatures, axis=1).values.astype(np.float32)
    labels = pdf['labels'].values.astype(np.int32)
    auxillary = pdf[['deltae', 'mbc']].values.astype(np.float32)

    return features, labels, auxillary, pdf

def dataset_placeholder(features_placeholder, labels_placeholder, batch_size, numEpochs, training=True, buffer_size=25600):
    dataset = tf.contrib.data.Dataset.from_tensor_slices((features_placeholder, labels_placeholder))
    dataset = dataset.shuffle(buffer_size=buffer_size)
    dataset = dataset.batch(batch_size)
    dataset = dataset.repeat(numEpochs) if training else dataset

    return dataset

def dataset_placeholder_aux(features_placeholder, labels_placeholder, auxillary_placeholder, batchSize, numEpochs, training=True, shuffle=True):
    dataset = tf.contrib.data.Dataset.from_tensor_slices((features_placeholder, labels_placeholder, auxillary_placeholder))
    if shuffle:
        dataset = dataset.shuffle(buffer_size=25600)
    dataset = dataset.batch(batchSize)
    dataset = dataset.repeat(numEpochs) if training else dataset

    return dataset

def dataset_placeholder_plot(features_placeholder, labels_placeholder, auxillary_placeholder, batchSize=512000, training=False):
    dataset = tf.contrib.data.Dataset.from_tensor_slices((features_placeholder, labels_placeholder, auxillary_placeholder))
    dataset = dataset.shuffle(buffer_size=batchSize)
    dataset = dataset.batch(batchSize)
    dataset = dataset.repeat()

    return dataset

def dataset_train(dataDirectory, batchSize, numEpochs, nFeatures, training=True):
    filenames = glob.glob('{}/part*'.format(dataDirectory))
    dataset = tf.contrib.data.TFRecordDataset(filenames)

    # Extract data from `tf.Example` protocol buffer
    def parser(record, batchSize=256):
        keys_to_features = {
            "features": tf.FixedLenFeature([nFeatures], tf.float32),
            "labels": tf.FixedLenFeature((), tf.float32,
            default_value=tf.zeros([], dtype=tf.float32)),
        }
        parsed = tf.parse_single_example(record, keys_to_features)
        label = tf.cast(parsed['labels'], tf.int32)

        return parsed['features'], label

    # Transform into feature, label tensor pair
    dataset = dataset.map(parser)
    dataset = dataset.shuffle(buffer_size=25600)
    dataset = dataset.batch(batchSize)
    dataset = dataset.repeat(numEpochs) if training else dataset

    return dataset

def dense_builder(x, shape, name, keep_prob, training=True, actv=tf.nn.elu):
    init=tf.contrib.layers.xavier_initializer()
    kwargs = {'center': True, 'scale': True, 'training': training, 'fused': True, 'renorm': True}

    with tf.variable_scope(name, initializer=init) as scope:
        layer = tf.layers.dense(x, units=shape[1], activation=actv, kernel_initializer=init)
        bn = tf.layers.batch_normalization(layer, **kwargs)
        layer_out = tf.layers.dropout(bn, 1-keep_prob, training=training)

    return layer_out

def selu_builder(x, shape, name, keep_prob, training=True):
    init = tf.contrib.layers.variance_scaling_initializer(factor=1.0, mode='FAN_IN')

    with tf.variable_scope(name) as scope:
        W = tf.get_variable("weights", shape = shape, initializer=init)
        b = tf.get_variable("biases", shape = [shape[1]], initializer=tf.random_normal_initializer(stddev=0.1))
        actv = selu.selu(tf.add(tf.matmul(x, W), b))
        layer_output = selu.dropout_selu(actv, rate=1-keep_prob, training=training)

    return layer_output

def dense_model(x, n_layers, hidden_layer_nodes, keep_prob, builder=selu_builder, reuse=False, training=True):
    # Extensible dense model
    SELU_initializer = tf.contrib.layers.variance_scaling_initializer(factor=1.0, mode='FAN_IN')
    init = SELU_initializer if builder==selu_builder else tf.contrib.layers.xavier_initializer()
    assert n_layers == len(hidden_layer_nodes), 'Specified layer nodes and number of layers do not correspond.'
    layers = [x]

    with tf.variable_scope('dense_model', reuse=reuse):
        hidden_0 = builder(x, shape=[config.nFeatures, hidden_layer_nodes[0]], name='hidden0',
                                keep_prob = keep_prob, training=training)
        layers.append(hidden_0)
        for n in range(0,n_layers-1):
            hidden_n = builder(layers[-1], shape=[hidden_layer_nodes[n], hidden_layer_nodes[n+1]], name='hidden{}'.format(n+1),
                                keep_prob=keep_prob, training=training)
            layers.append(hidden_n)

        readout = tf.layers.dense(hidden_n, units=config.n_classes, kernel_initializer=init)

    return readout

def custom_elu(x):
    return tf.nn.elu(x) + 1.0

def log_sum_exp_trick(x, axis=1):
    x_max = tf.reduce_max(x, axis=1, keep_dims=True)
    lse = x_max + tf.log(tf.reduce_sum(tf.exp(x-x_max), axis=1, keep_dims=True))
    return lse

def scope_variables(name):
    with tf.variable_scope(name):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                       scope=tf.get_variable_scope().name)

def adversary_dense(x, z_pivot, n_layers, hidden_nodes, keep_prob, name, builder=selu_builder, reuse=False, training=True, actv=selu.selu):
    # Extensible dense model
    SELU_initializer = tf.contrib.layers.variance_scaling_initializer(factor=1.0, mode='FAN_IN')
    init = SELU_initializer if builder==selu_builder else tf.contrib.layers.xavier_initializer()
    assert n_layers == len(hidden_nodes), 'Specified layer nodes and number of layers do not correspond.'
    layers = [x]

    with tf.variable_scope('adversary', reuse=reuse):
        hidden_0 = builder(x, shape=[2, hidden_nodes[0]], name='hidden0',
                                keep_prob=keep_prob, training=training)
        layers.append(hidden_0)
        for n in range(0,n_layers-1):
            hidden_n = builder(layers[-1], shape=[hidden_nodes[n], hidden_nodes[n+1]], name='hidden{}'.format(n+1),
                                keep_prob=keep_prob, training=training)
            layers.append(hidden_n)

        fc = tf.layers.dense(hidden_n, units=96, activation=tf.nn.tanh, kernel_initializer=init)
        fc_logits, fc_mu, fc_sigma = tf.split(fc, 3, axis=1)
        logits = tf.layers.dense(fc_logits, units=config.n_gaussians, activation=tf.identity, name='mixing_fractions')
        centers = tf.layers.dense(fc_mu, units=config.n_gaussians, activation=tf.identity, name='means')
        variances = tf.layers.dense(fc_sigma, units=config.n_gaussians, activation=custom_elu, name='variances')
        mixing_coeffs = tf.nn.softmax(logits)

        exponent = tf.log(mixing_coeffs) - 1/2 * tf.log(2*np.pi) - tf.log(variances) - tf.square(centers-tf.expand_dims(z_pivot,1))/(2*tf.square(variances))

    return log_sum_exp_trick(exponent)

def selu_adversary(x, z_pivot, n_layers, hidden_nodes, keep_prob, name, reuse=False, training=True, actv=selu.selu):
    SELU_initializer = tf.contrib.layers.variance_scaling_initializer(factor=1.0, mode='FAN_IN')
    xavier = tf.contrib.layers.xavier_initializer(uniform=True, seed=None, dtype=tf.float32)

    with tf.variable_scope('adversary', reuse=reuse):
        l0 = tf.layers.dense(x, units=hidden_nodes[0], activation=actv, kernel_initializer=SELU_initializer)
        d0 = selu.dropout_selu(l0, rate=1-keep_prob, training=training)

        l1 = tf.layers.dense(d0, units=hidden_nodes[1], activation=actv, kernel_initializer=SELU_initializer)
        d1 = selu.dropout_selu(l1, rate=1-keep_prob, training=training)

        fc = tf.layers.dense(d1, units=96, activation=tf.nn.tanh, kernel_initializer=xavier)
        fc_logits, fc_mu, fc_sigma = tf.split(fc, 3, axis=1)
        logits = tf.layers.dense(fc_logits, units=config.n_gaussians, activation=tf.identity, name='mixing_fractions')
        centers = tf.layers.dense(fc_mu, units=config.n_gaussians, activation=tf.identity, name='means')
        variances = tf.layers.dense(fc_sigma, units=config.n_gaussians, activation=custom_elu, name='variances')
        mixing_coeffs = tf.nn.softmax(logits)

        exponent = tf.log(mixing_coeffs) - 1/2 * tf.log(2*np.pi) - tf.log(variances) - tf.square(centers-tf.expand_dims(z_pivot,1))/(2*tf.square(variances))

    return log_sum_exp_trick(exponent)


class vanillaDNN():
    # Builds the computational graph
    def __init__(self, config, training=True, cyclical=False):

        self.global_step = tf.Variable(0, trainable=False)
        self.handle = tf.placeholder(tf.string, shape=[])
        self.training_phase = tf.placeholder(tf.bool)
        self.beta = tf.placeholder(tf.float32) if cyclical else config.learning_rate
        self.features_placeholder = tf.placeholder(features.dtype, features.shape)
        self.labels_placeholder = tf.placeholder(labels.dtype, labels.shape)
        self.auxillary_placeholder = tf.placeholder(auxillary.dtype, auxillary.shape)
        self.featuresTest_placeholder = tf.placeholder(features.dtype, featuresTest.shape)
        self.labelsTest_placeholder = tf.placeholder(labels.dtype, labelsTest.shape)
        self.auxillaryTest_placeholder = tf.placeholder(auxillary.dtype, auxillaryTest.shape)

        trainDataset = dataset_placeholder_aux(self.features_placeholder, self.labels_placeholder, self.auxillary_placeholder,
                                           config.batch_size, config.num_epochs, training=True)
        testDataset = dataset_placeholder_aux(self.featuresTest_placeholder, self.labelsTest_placeholder, self.auxillaryTest_placeholder,
                                          config.batch_size, config.num_epochs, training=True)
        plotDataset = dataset_placeholder_plot(self.featuresTest_placeholder, self.labelsTest_placeholder, self.auxillaryTest_placeholder)

        self.iterator = tf.contrib.data.Iterator.from_string_handle(self.handle, trainDataset.output_types, trainDataset.output_shapes)
        self.train_iterator = trainDataset.make_initializable_iterator()
        self.test_iterator = testDataset.make_initializable_iterator()
        self.plot_iterator = plotDataset.make_initializable_iterator()

        self.example, self.label, self.ancillary = self.iterator.get_next()
        self.readout = dense_model(self.example, config.n_layers, config.hidden_layer_nodes, config.keep_prob, builder=selu_builder,
                                   reuse=False, training=self.training_phase)
        # Introduce a separate adversary network for each pivot
        self.gmm_log_ll = adversary_dense(tf.nn.softmax(self.readout), self.ancillary[:,0], config.adv_n_layers, config.adv_hidden_nodes,
                config.adv_keep_prob, builder=dense_builder, reuse=False, training=self.training_phase, name='deltaE')

        # Mask signal values in adversary loss
        self.adversary_loss = -tf.reduce_mean(tf.cast((1-self.label), tf.float32)*self.gmm_log_ll)
        self.predictor_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.readout, labels=self.label))
        self.total_loss = self.predictor_loss - config.adv_lambda*self.adversary_loss

        theta_f = scope_variables('dense_model')
        theta_r = scope_variables('adversary')

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            # Ensures that we execute the update_ops before performing the train_step
            predictor_optimizer = tf.train.AdamOptimizer(config.learning_rate)
            predictor_gs = tf.Variable(0, name='predictor_global_step', trainable=False)
            self.predictor_train_op = predictor_optimizer.minimize(self.predictor_loss, name='predictor_opt', global_step=predictor_gs, var_list=theta_f)
            predictor_optimize = predictor_optimizer.minimize(self.total_loss, name='predictor_opt', global_step=predictor_gs, var_list=theta_f)
            # self.joint_train_op = predictor_optimizer.minimize(self.total_loss, name='joint_opt', global_step=predictor_gs, var_list=theta_f)

            adversary_optimizer = tf.train.AdamOptimizer(config.adv_learning_rate)
            adversary_gs = tf.Variable(0, name='adversary_global_step', trainable=False)
            self.adversary_train_op = adversary_optimizer.minimize(self.adversary_loss, name='adversary_opt', global_step=adversary_gs, var_list=theta_r)

        self.ema = tf.train.ExponentialMovingAverage(decay=config.ema_decay, num_updates=predictor_gs, name='predictor_ema')
        maintain_predictor_averages_op = self.ema.apply(theta_f)
        with tf.control_dependencies([predictor_optimize]):
            self.joint_train_op = tf.group(maintain_predictor_averages_op)

        # Evaluation metrics
        self.cross_entropy = self.predictor_loss
        self.p = tf.nn.softmax(self.readout)
        self.transform = tf.log(self.p[:,1]/(1-self.p[:,1]+config.epsilon)+config.epsilon)
        correct_prediction = tf.equal(tf.cast(tf.argmax(self.readout, 1), tf.int32), self.label)
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        _, self.auc_op = tf.metrics.auc(predictions = tf.argmax(self.readout,1), labels = self.label, num_thresholds = 1024)
        self.pearson_dE, self.pearson_dE_op =  tf.contrib.metrics.streaming_pearson_correlation(predictions=self.transform,
                                                                                                labels=self.ancillary[:,0], name='pearson_dE')
        self.pearson_mbc, self.pearson_mbc_op =  tf.contrib.metrics.streaming_pearson_correlation(predictions=self.transform,
                                                                                                  labels=self.ancillary[:,1], name='pearson_mbc')
        # Add summaries
        tf.summary.scalar('accuracy', self.accuracy)
        tf.summary.scalar('auc', self.auc_op)
        tf.summary.scalar('learning_rate', self.beta)
        tf.summary.scalar('predictor_loss', self.predictor_loss)
        tf.summary.scalar('adversary_loss', self.adversary_loss)
        tf.summary.scalar('total_loss', self.total_loss)
        tf.summary.scalar('pearson_dE', self.pearson_dE_op)
        tf.summary.scalar('pearson_mbc', self.pearson_mbc_op)

        self.merge_op = tf.summary.merge_all()
        self.train_writer = tf.summary.FileWriter(
            os.path.join(directories.tensorboard, 'train_{}'.format(time.strftime('%d-%m_%I:%M'))), graph = tf.get_default_graph())
        self.test_writer = tf.summary.FileWriter(
            os.path.join(directories.tensorboard, 'test_{}'.format(time.strftime('%d-%m_%I:%M'))))

    def predict(self, ckpt):
        pin_cpu = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True, device_count = {'GPU':0})
        start_time = time.time()

        # Restore the moving average version of the learned variables for eval.
        #variables_to_restore = self.predictor_ema.variables_to_restore() + self.adversary_ema.variables_to_restore()
        #saver = tf.train.Saver(variables_to_restore)
        saver = tf.train.Saver()
        valDataset = dataset_placeholder_aux(self.featuresTest_placeholder, self.labelsTest_placeholder, self.auxillaryTest_placeholder,
                                          config.batch_size, config.num_epochs, training=False, shuffle=False)
        val_iterator = valDataset.make_initializable_iterator()
        concatLabels = tf.cast(self.label, tf.int32)
        concatPreds = tf.cast(tf.argmax(self.readout,1), tf.int32)
        concatOutput = self.p[:,1]

        with tf.Session(config=pin_cpu) as sess:
            # Initialize variables
            init_op = tf.global_variables_initializer()
            sess.run(init_op)
            sess.run(tf.local_variables_initializer())
            assert (ckpt.model_checkpoint_path), 'Missing checkpoint file!'
            saver.restore(sess, ckpt.model_checkpoint_path)
            print('{} restored.'.format(ckpt.model_checkpoint_path))
            val_handle = sess.run(val_iterator.string_handle())
            labels, preds, outputs = [], [], []
            sess.run(val_iterator.initializer, feed_dict={vDNN.featuresTest_placeholder: featuresTest,
                                                          vDNN.labelsTest_placeholder: labelsTest,
                                                          vDNN.auxillaryTest_placeholder: auxillaryTest})
            while True:
                try:
                    l, p, o = sess.run([concatLabels, concatPreds, concatOutput],
                                       feed_dict = {vDNN.training_phase: False, vDNN.handle: val_handle})
                    labels.append(l), preds.append(p), outputs.append(o)
                except tf.errors.OutOfRangeError:
                    labels, preds, outputs = np.concatenate(labels), np.concatenate(preds), np.concatenate(outputs)
                    break
            acc = np.mean(np.equal(labels,preds))
            print("Validation accuracy: {:.3f}".format(acc))

            plot_ROC_curve(network_output=outputs, y_true=labels, identifier=config.mode+config.channel,
                           meta=architecture + ' | Test accuracy: {:.3f}'.format(acc))
            delta_t = time.time() - start_time
            print("Inference complete. Duration: %g s" %(delta_t))

            return labels, preds, outputs


def train(config, restore = False):
    # Executes training operations
    print('Architecture: {}'.format(architecture))
    vDNN = vanillaDNN(config, training=True)
    start_time = time.time()
    global_step, n_checkpoints, v_auc_best = 0, 0, 0.
    saver = tf.train.Saver()
    ckpt = tf.train.get_checkpoint_state(directories.checkpoints)

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)) as sess:
        # Initialize variables
        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        train_handle = sess.run(vDNN.train_iterator.string_handle())
        test_handle = sess.run(vDNN.test_iterator.string_handle())
        plot_handle = sess.run(vDNN.plot_iterator.string_handle())

        if restore and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            print('{} restored.'.format(ckpt.model_checkpoint_path))

        sess.run(vDNN.train_iterator.initializer, feed_dict={vDNN.features_placeholder: features,
                                                             vDNN.labels_placeholder: labels,
                                                             vDNN.auxillary_placeholder: auxillary})
        sess.run(vDNN.test_iterator.initializer, feed_dict={vDNN.featuresTest_placeholder: featuresTest,
                                                            vDNN.labelsTest_placeholder: labelsTest,
                                                            vDNN.auxillaryTest_placeholder: auxillaryTest})
        sess.run(vDNN.plot_iterator.initializer, feed_dict={vDNN.featuresTest_placeholder: featuresTest,
                                                            vDNN.labelsTest_placeholder: labelsTest,
                                                            vDNN.auxillaryTest_placeholder: auxillaryTest})
        while True:
            try:
                if config.adversary:
                    # adversary trains in inner loop
                    if global_step % config.K == 0:
                        sess.run(vDNN.joint_train_op, feed_dict={vDNN.training_phase: True, vDNN.handle: train_handle})
                    else:
                        sess.run(vDNN.adversary_train_op, feed_dict={vDNN.training_phase: True, vDNN.handle: train_handle})
                    global_step+=1

                    if global_step % (config.steps_per_epoch) == 0:
                        epoch, v_auc_best = run_adv_diagnostics(vDNN, config, directories, sess, saver, train_handle, test_handle,
                                                            global_step, config.nTrainExamples, start_time, v_auc_best, n_checkpoints)
                        plot_distributions(vDNN, epoch, sess, handle=plot_handle, notebook=False)
                else:
                    # Run X steps on training dataset
                    sess.run(vDNN.predictor_train_op, feed_dict={vDNN.training_phase: True, vDNN.handle: train_handle})
                    global_step+=1

                    if global_step % (config.steps_per_epoch // 4) == 0:
                        epoch, v_auc_best = run_diagnostics(vDNN, config, directories, sess, saver, train_handle, test_handle,
                                                            global_step, config.nTrainExamples, start_time, v_auc_best, n_checkpoints)

            except tf.errors.OutOfRangeError:
                break

        save_path = saver.save(sess, os.path.join(directories.checkpoints, 'vDNN_{}_{}_end.ckpt'.format(config.mode, config.channel)),
                               global_step=epoch)

    print("Training Complete. Model saved to file: {} Time elapsed: {:.3f} s".format(save_path, time.time()-start_time))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--predict', help = 'Run inference', action = 'store_true')
    parser.add_argument('-n', '--n_epochs', type = int,  help = 'Number of training epochs')
    args = parser.parse_args()

    if args.n_epochs:
        config.num_epochs = args.n_epochs

    get_available_gpus()
    features, labels, auxillary, pdf = load_parquet(directories.train)
    featuresTest, labelsTest, auxillaryTest, pdf_test = load_parquet(directories.test)
    config.nTrainExamples, config.nFeatures = features.shape[0], features.shape[-1]
    config.steps_per_epoch = features.shape[0] // config.batch_size

    architecture = '{} - {} | Layers: {} | Dropout: {} | Base LR: {} | Epochs: {}'.format(
        config.channel, config.mode, config.n_layers, config.keep_prob, config.learning_rate, config.num_epochs)

    if args.predict:
        ckpt = tf.train.get_checkpoint_state(directories.checkpoints)
        vDNN = vanillaDNN(config, training = False)
        labels, preds, output = vDNN.predict(ckpt)
        # Add predictions to test set as a new column, save as HDF5
        test = pd.concat([pdf_test, pd.Series(output, name='probs')], axis=1)
        test.to_hdf(os.path.join(directories.checkpoints, '{}_preds.h5'.format(os.path.basename(directories.test))), key = 'df', format='t', data_columns=True)
    else:
        # Periodically saves graphs to check for distribution sculpting
        train(config)

