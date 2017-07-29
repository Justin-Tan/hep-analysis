
# coding: utf-8

# ## vanilla-DNN-ema

# Author: Justin Tan
# 
# Vanilla neural network. Do anything from MNIST to signal classification.
# 
# Update 20/03: Added batch normalization, TensorBoard visualization
# 
# Update 19/06: Added cosine annealing, exponential moving average
# 
# To-do: Update to TF 1.2, fused batch norm

# In[1]:

import tensorflow as tf
import numpy as np
import pandas as pd
import time, os

class config(object):
    # Set network parameters
    mode = 'kst'
    channel = 'rho0'
    n_features = 100
    keep_prob = 0.8
    num_epochs = 256
    batch_size = 256
#     n_layers = 3
#     hidden_layer_nodes = [1024, 1024, 512]
    n_layers = 12
    hidden_layer_nodes = [1024, 1024, 1024, 1024, 512, 512, 512, 512, 256, 256, 256, 256]
    ema_decay = 0.999
    learning_rate = 8e-5
    cycles = 8 # Number of annealing cycles
    n_classes = 2
    builder = 'selu'

class directories(object):
    tensorboard = 'tensorboard'
    checkpoints = 'checkpoints'
    
architecture = '{} - {} | Layers: {} | Dropout: {} | Base LR: {} | Epochs: {}'.format(
    config.channel, config.mode, config.n_layers, config.keep_prob, config.learning_rate, config.num_epochs)

class reader():
    # Iterates over data and returns batches
    def __init__(self, df):
        
        self.df = df
        self.batch_size = config.batch_size
        self.steps_per_epoch = len(df) // config.batch_size
        self.epochs = 0
        self.proceed = True
        self.shuffle()

    def shuffle(self):
        self.df = self.df.sample(frac=1).reset_index(drop=True)
        self.df_X = self.df.drop('labels', axis = 1)
        self.df_y = self.df['labels']
        self.pointer = 0


    def next_batch(self, batch_size):
        if self.pointer + 1 >= self.steps_per_epoch:
            inputs = self.df_X.iloc[self.pointer*batch_size:]
            targets = self.df_y.iloc[self.pointer*batch_size:]
            self.epochs += 1
            self.shuffle()
            self.proceed = False
            
        inputs = self.df_X.iloc[self.pointer*batch_size:(self.pointer+1)*batch_size]
        targets = self.df_y.iloc[self.pointer*batch_size:(self.pointer+1)*batch_size]
        self.pointer += 1
                
        return inputs, targets


# ### Functions for graph construction

# In[2]:

# SELU helper functions
import numbers
from tensorflow.contrib import layers
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.layers import utils

""" When using SELUs you have to keep the following in mind:
# (1) scale inputs to zero mean and unit variance
# (2) use SELUs
# (3) initialize weights with stddev sqrt(1/n)
# (4) use SELU dropout
"""
    
# (1) scale inputs to zero mean and unit variance

# (2) use SELUs
def selu(x):
    with ops.name_scope('elu') as scope:
        alpha = 1.6732632423543772848170429916717
        scale = 1.0507009873554804934193349852946
        return scale*tf.where(x>=0.0, x, alpha*tf.nn.elu(x))
    
# (3) initialize weights with stddev sqrt(1/n)
SELU_initializer = layers.variance_scaling_initializer(factor=1.0, mode='FAN_IN')

# (4) use this dropout
def dropout_selu(x, rate, alpha= -1.7580993408473766, fixedPointMean=0.0, fixedPointVar=1.0,
                 noise_shape=None, seed=None, name=None, training=False):
    """Dropout to a value with rescaling."""

    def dropout_selu_impl(x, rate, alpha, noise_shape, seed, name):
        keep_prob = 1.0 - rate
        x = ops.convert_to_tensor(x, name="x")
        if isinstance(keep_prob, numbers.Real) and not 0 < keep_prob <= 1:
            raise ValueError("keep_prob must be a scalar tensor or a float in the "
                                             "range (0, 1], got %g" % keep_prob)
        keep_prob = ops.convert_to_tensor(keep_prob, dtype=x.dtype, name="keep_prob")
        keep_prob.get_shape().assert_is_compatible_with(tensor_shape.scalar())

        alpha = ops.convert_to_tensor(alpha, dtype=x.dtype, name="alpha")
        alpha.get_shape().assert_is_compatible_with(tensor_shape.scalar())

        if tensor_util.constant_value(keep_prob) == 1:
            return x

        noise_shape = noise_shape if noise_shape is not None else array_ops.shape(x)
        random_tensor = keep_prob
        random_tensor += random_ops.random_uniform(noise_shape, seed=seed, dtype=x.dtype)
        binary_tensor = math_ops.floor(random_tensor)
        ret = x * binary_tensor + alpha * (1-binary_tensor)

        a = math_ops.sqrt(fixedPointVar / (keep_prob *((1-keep_prob) * math_ops.pow(alpha-fixedPointMean,2) + fixedPointVar)))

        b = fixedPointMean - a * (keep_prob * fixedPointMean + (1 - keep_prob) * alpha)
        ret = a * ret + b
        ret.set_shape(x.get_shape())
        return ret

    with ops.name_scope(name, "dropout", [x]) as name:
        return utils.smart_cond(training,
                                lambda: dropout_selu_impl(x, rate, alpha, noise_shape, seed, name),
                                lambda: array_ops.identity(x))


# In[3]:

def load_data(file_name, test_size = 0.05):
    from sklearn.model_selection import train_test_split
    df = pd.read_hdf(file_name, 'df')
    df_X_train, df_X_test, df_y_train, df_y_test = train_test_split(df.drop('labels', axis = 1),
                                                                    df['labels'], test_size = test_size, random_state=42)
    return df_X_train, df_X_test, df_y_train, df_y_test

def save_summary(config, delta_t, train_acc, test_acc, test_auc):
    import json
    summary = {
        'Channel': config.channel,
        'Mode': config.mode,
        'Timestamp': time.strftime('%c'),
        'Arch': config.builder,
        'Layers': config.n_layers,
        'Batch_size': config.batch_size,
        'Dropout': config.keep_prob,
        'Epochs': config.num_epochs,
        'Time': delta_t,
        'Final train acc': train_acc,
        'Final test acc': test_acc,
        'Final test AUC': test_auc
    }
    # Writing JSON data
    if os.path.isfile('vdnn_summary.json'):
        with open('vdnn_summary.json.', 'r+') as f:
            new = json.load(f)
        new.append(summary)
        with open('vdnn_summary.json', 'w') as f:
            json.dump(new, f, indent = 4)
    else:
        with open('vdnn_summary.json', 'w') as f:
             json.dump([summary], f, indent = 4)

def layer_weights(shape, initializer = tf.contrib.layers.xavier_initializer()):
    # Return weight tensor of given shape using Xavier initialization
    W = tf.get_variable("weights", shape = shape, initializer=initializer)
    return W

def layer_biases(shape, init_value = 0.0):
    # Return bias tensor of given shape with small initialized constant value
    b = tf.get_variable("biases", shape = shape, initializer = tf.constant_initializer(init_value))
    return b

def hidden_layer_ops(x, shape, name, keep_prob, activation=tf.nn.relu):
    # Add operations to graph to construct hidden layers
    with tf.variable_scope(name) as scope:
        # scope.reuse_variables() # otherwise tf.get_variable() checks that already existing vars are not shared by accident
        weights = layer_weights(shape = shape)
        biases = layer_biases(shape = [shape[1]])
        
        # Apply non-linearity. Default is ReLU
        actv = activation(tf.matmul(x, weights) + biases)
        layer_output = tf.nn.dropout(actv, keep_prob)
        
    return layer_output

def hidden_SELU_ops(x, shape, name, keep_prob, phase = True):
    # Add operations to graph to construct hidden layers
    with tf.variable_scope(name) as scope:
        # scope.reuse_variables() # otherwise tf.get_variable() checks that already existing vars are not shared by accident
        weights = layer_weights(shape = shape, initializer = SELU_initializer)
        biases = layer_biases(shape = [shape[1]])
        
        # Apply non-linearity. Default is ReLU
        actv = selu(tf.add(tf.matmul(x, weights), biases))
        layer_output = dropout_selu(actv, rate = 1 - keep_prob, training = phase)
        
    return layer_output

def build_SELU_network2(x, n_layers, hidden_layer_nodes, keep_prob, training_phase):
    shape = [config.n_features, 1024]
    name = 'w1'
    with tf.variable_scope(name) as scope:
        w1 = layer_weights(shape = shape, initializer = SELU_initializer)
        b1 = layer_biases(shape = [shape[1]])

    # Apply non-linearity. Default is ReLU
    a1 = selu(tf.add(tf.matmul(x, w1), b1))
    h1 = dropout_selu(a1, rate = 1 - keep_prob, training = training_phase)
    
    shape = [1024, 1024]
    name = 'w2'
    with tf.variable_scope(name) as scope:
        w2 = layer_weights(shape = shape, initializer = SELU_initializer)
        b2 = layer_biases(shape = [shape[1]])
    
    # Apply non-linearity. Default is ReLU
    a2 = selu(tf.add(tf.matmul(h1, w2), b2))
    h2 = dropout_selu(a2, rate = 1 - keep_prob, training = training_phase)
    
    name = 'w3'
    shape = [1024, 512]

    with tf.variable_scope(name) as scope:
        w3 = layer_weights(shape = shape, initializer = SELU_initializer)
        b3 = layer_biases(shape = [shape[1]])
    
    # Apply non-linearity. Default is ReLU
    a3 = selu(tf.add(tf.matmul(h2, w3), b3))
    h3 = dropout_selu(a3, rate = 1 - keep_prob, training = training_phase)
    
    shape = [512, 2]
    name = 'w4'
    with tf.variable_scope(name) as scope:
        w4 = layer_weights(shape = shape, initializer = SELU_initializer)
        b4 = layer_biases(shape = [shape[1]])
    
    # Apply non-linearity. Default is ReLU
    readout = tf.add(tf.matmul(h3, w4), b4)
    return readout

def readout_ops(x, shape, name, initializer = tf.contrib.layers.xavier_initializer()):
    # Don't apply non-linearity, dropout on output layer
    with tf.variable_scope(name) as scope:
        weights = layer_weights(shape = shape, initializer = initializer)
        biases = layer_biases(shape = [shape[1]])
        layer_output = tf.matmul(x, weights) + biases
        
    return layer_output

def BN_layer_ops(x, shape, name, keep_prob, phase, activation=tf.nn.relu):
    # High-level implementation of BN
    with tf.variable_scope(name) as scope:
         # scope.reuse_variables() # otherwise tf.get_variable() checks that already existing vars are not shared by accident
        weights = layer_weights(shape = shape)
        biases = layer_biases(shape = [shape[1]])
        z_BN = tf.matmul(x, weights) + biases
        
        # Place BN transform before non-linearity - update to TF 1.2!
        theta_BN = tf.contrib.layers.batch_norm(z_BN, center=True, scale=True,is_training=phase, 
                                                decay=0.99, zero_debias_moving_mean=True, scope='bn', fused = True)
        BN_actv = activation(theta_BN)
        BN_layer_output = tf.nn.dropout(BN_actv, keep_prob)

    return BN_layer_output

def SELU_BN_layer_ops(x, shape, name, keep_prob, phase):
    # High-level implementation of BN
    with tf.variable_scope(name) as scope:
         # scope.reuse_variables() # otherwise tf.get_variable() checks that already existing vars are not shared by accident
        weights = layer_weights(shape = shape)
        biases = layer_biases(shape = [shape[1]])
        z_BN = tf.matmul(x, weights) + biases
        
        # Place BN transform before non-linearity - update to TF 1.2!
        theta_BN = tf.contrib.layers.batch_norm(z_BN, center=True, scale=True,is_training=phase, 
                                                decay=0.99, zero_debias_moving_mean=True, scope='bn', fused = True)
        BN_actv = selu(theta_BN)
        BN_layer_output = dropout_selu(BN_actv, rate = 1 - keep_prob, training = phase)

    return BN_layer_output

def network_builder(x, n_layers, hidden_layer_nodes, keep_prob, training_phase):
    assert n_layers == len(hidden_layer_nodes), 'Specified layer nodes and number of layers do not correspond.'
    layers = [x]
    if config.builder == 'bn':
        print('Building ReLU + Batch-norm architecture')
        builder = BN_layer_ops
    elif config.builder == 'selu':
        print('Building SELU architecture')
        builder = hidden_SELU_ops
    elif config.builder == 'selu-bn':
        print('Building SELU + Batch-norm architecture')
        builder = SELU_BN_layer_ops
    else:
        print('Default architecture: SELU')
        builder = hidden_SELU_ops
        
    with tf.variable_scope('hidden_layers') as scope:
        hidden_1 = builder(x, shape = [config.n_features, hidden_layer_nodes[0]], name = 'hidden0',
                                keep_prob = keep_prob, phase = training_phase)
        layers.append(hidden_1)
        for n in range(0,n_layers-1):
            hidden_n = builder(layers[-1], shape = [hidden_layer_nodes[n], hidden_layer_nodes[n+1]], name = 'hidden{}'.format(n+1),
                                   keep_prob = keep_prob, phase = training_phase)
            layers.append(hidden_n)
        readout = readout_ops(layers[-1], shape = [hidden_layer_nodes[-1], config.n_classes], name = 'readout', initializer = SELU_initializer)
        
        return readout

def build_SELU_network(x, n_layers, hidden_layer_nodes, keep_prob, training_phase):
    assert n_layers == len(hidden_layer_nodes), 'Specified layer nodes and number of layers do not correspond.'
    layers = [x]
    with tf.variable_scope('SELU_layers') as scope:
        hidden_1 = hidden_SELU_ops(x, shape = [config.n_features, hidden_layer_nodes[0]], name = 'SELUhidden0',
                                keep_prob = keep_prob, phase = training_phase)
        layers.append(hidden_1)
        for n in range(0,n_layers-1):
            hidden_n = hidden_SELU_ops(layers[-1], shape = [hidden_layer_nodes[n], hidden_layer_nodes[n+1]], name = 'SELUhidden{}'.format(n+1),
                                   keep_prob = keep_prob, phase = training_phase)
            layers.append(hidden_n)
        readout = readout_ops(layers[-1], shape = [hidden_layer_nodes[-1], config.n_classes], name = 'readout', initializer = SELU_initializer)
        
        return readout

def build_network(x, n_layers, hidden_layer_nodes, keep_prob, training_phase):
    assert n_layers == len(hidden_layer_nodes), 'Specified layer nodes and number of layers do not correspond.'
    layers = [x]
    with tf.variable_scope('BN_layers') as scope:
        hidden_1 = BN_layer_ops(x, shape = [config.n_features, hidden_layer_nodes[0]], name = 'BNhidden0',
                                keep_prob = keep_prob, phase = training_phase)
        layers.append(hidden_1)
        for n in range(0,n_layers-1):
            hidden_n = BN_layer_ops(layers[-1], shape = [hidden_layer_nodes[n], hidden_layer_nodes[n+1]], name = 'BNhidden{}'.format(n+1),
                                   keep_prob = keep_prob, phase = training_phase)
            layers.append(hidden_n)
        readout = readout_ops(layers[-1], shape = [hidden_layer_nodes[-1], config.n_classes], name = 'readout')
        
        return readout

def plot_ROC_curve(network_output, y_true, meta = ''):
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import roc_curve, auc

    y_score = network_output[:,1]
    
    # Compute ROC curve, integrate
    fpr, tpr, thresholds = roc_curve(y_true, y_score)    
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
    plt.savefig(os.path.join('graphs', '{}_{}_ROC.pdf'.format(config.channel, config.mode)), format='pdf', dpi=1000)
    plt.show()
    plt.gcf().clear()
    
def cosine_anneal(initial_lr, t, T, M):
    from math import ceil
    beta = initial_lr/2 * (np.cos(np.pi* (t % ceil(T/M))/ceil(T/M)) + 1)
    return beta


# In[4]:

test_file = '/data/projects/punim0011/jtan/data/dnn/norm_dnn_train_B2rho0gamma_kst.h5'

df_X_train, df_X_test, df_y_train, df_y_test = load_data(test_file)
df_y_train = df_y_train.astype(np.int8)
df_y_test = df_y_test.astype(np.int8)
df_train = pd.concat([df_X_train, df_y_train], axis = 1)
df_test = pd.concat([df_X_test, df_y_test], axis = 1)
config.n_features = df_train.shape[1]-1
config.steps_per_epoch = df_train.shape[0] // config.batch_size
config.T = config.steps_per_epoch*config.num_epochs

readerTrain = reader(df_train)
readerTest = reader(df_test)


# In[5]:

class vanillaDNN():
    # Builds the computational graph
    def __init__(self, config, training = True, cyclical = False):
        
        self.x = tf.placeholder(tf.float32, shape = [None, config.n_features])
        self.y_true = tf.placeholder(tf.int32, shape = None)
        self.keep_prob = tf.placeholder(tf.float32)
        self.training_phase = tf.placeholder(tf.bool)
        if cyclical:
            self.beta = tf.placeholder(tf.float32)
        else:
            self.beta = config.learning_rate

        # Anneal learning rate
        self.global_step = tf.Variable(0, trainable=False)
#         self.beta = tf.train.exponential_decay(config.learning_rate, self.global_step, 
#                                                decay_steps = config.steps_per_epoch, decay_rate = config.lr_epoch_decay, staircase=True)
        
#         if config.builder == 'selu':
#             print('Using SELU activation')
#             self.readout = build_SELU_network(self.x, config.n_layers, config.hidden_layer_nodes, self.keep_prob, self.training_phase)
#             self.cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits = self.readout, labels = self.y_true))
#             self.opt_op = tf.train.AdamOptimizer(self.beta).minimize(self.cross_entropy, name = 'optimizer',
#                                                                      global_step = self.global_step)
         
        self.readout = network_builder(self.x, config.n_layers, config.hidden_layer_nodes, self.keep_prob, self.training_phase)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        self.cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits = self.readout, labels = self.y_true))

        with tf.control_dependencies(update_ops):
            # Ensures that we execute the update_ops before performing the train_step
            self.opt_op = tf.train.AdamOptimizer(self.beta).minimize(self.cross_entropy, name = 'optimizer',
                                                                     global_step = self.global_step)


        ema = tf.train.ExponentialMovingAverage(decay = config.ema_decay, num_updates = self.global_step)
        maintain_averages_op = ema.apply(tf.trainable_variables())
        
        with tf.control_dependencies([self.opt_op]):
            self.train_op = tf.group(maintain_averages_op)

        # Evaluation metrics
        self.prediction = tf.nn.softmax(self.readout)
        correct_prediction = tf.equal(tf.cast(tf.argmax(self.readout, 1), tf.int32), self.y_true)
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        _, self.auc_op = tf.metrics.auc(predictions = tf.argmax(self.readout,1), labels = self.y_true, num_thresholds = 1024)
        tf.summary.scalar('accuracy', self.accuracy)
        tf.summary.scalar('auc', self.auc_op)
        tf.summary.scalar('learning_rate', self.beta)
        tf.summary.scalar('cross_entropy', self.cross_entropy)
        
        self.merge_op = tf.summary.merge_all()
        self.train_writer = tf.summary.FileWriter(
            os.path.join(directories.tensorboard, 'train_{}'.format(time.strftime('%d-%m_%I:%M'))), graph = tf.get_default_graph())
        self.test_writer = tf.summary.FileWriter(
            os.path.join(directories.tensorboard, 'test_{}'.format(time.strftime('%d-%m_%I:%M'))))
        
    def predict(self, ckpt, metaGraph = None):
        # Restore the moving average version of the learned variables for eval.
        variable_averages = tf.train.ExponentialMovingAverage(config.ema_decay)
        variables_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)
        
        with tf.Session() as sess:
            # Initialize variables
            init_op = tf.global_variables_initializer()
            sess.run(init_op)
            sess.run(tf.local_variables_initializer())

            start_time = time.time()
            
            assert (ckpt.model_checkpoint_path or metaGraph), 'Missing checkpoint file!'
            
            if metaGraph:
                saver = tf.train.import_meta_graph(metaGraph)
                saver.restore(sess, os.path.splitext(metaGraph)[0])
                print('{} restored.'.format(metaGraph))
            else:    
                saver.restore(sess, ckpt.model_checkpoint_path)
                print('{} restored.'.format(ckpt.model_checkpoint_path))

            # Make predictions using the trained model
            feed_dict_test = {self.x: df_X_test.values, self.y_true: df_y_test.values, self.keep_prob: 1.0, self.training_phase: False}
            network_output_test, final_v_acc, final_v_auc = sess.run(
                [self.prediction, self.accuracy, self.auc_op], feed_dict = feed_dict_test)

            print("Validation accuracy: {:g}\nValidation AUC: {:g}".format(final_v_acc, final_v_auc))
            
            plot_ROC_curve(network_output = network_output_test, y_true = df_y_test.values, meta = architecture)
            delta_t = time.time() - start_time
            print("Inference complete. Duration: %g s" %(delta_t))
            
            return network_output_test


# In[6]:

def train(config, restore = False):
    # Executes training operations
    
    vDNN = vanillaDNN(config, training = True)
    start_time = time.time()
    v_acc_best = 0.
    global_step = 0
    saver = tf.train.Saver()
    ckpt = tf.train.get_checkpoint_state(directories.checkpoints)
    
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)) as sess:
        # Initialize variables
        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        
        if restore and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            print('{} restored.'.format(ckpt.model_checkpoint_path))
            
        for epoch in range(config.num_epochs):
            
            readerTrain.proceed = True
            step = 0
            # Save every 10 epochs    
            if epoch % 10 == 0:
                save_path = saver.save(sess,
                                       os.path.join(directories.checkpoints,
                                                    'vDNN_{}_{}_epoch{}.ckpt'.format(config.mode, config.channel, epoch)),
                                       global_step = epoch)
                print('Graph saved to file: {}'.format(save_path))
                
            print('(*) Entering Epoch {} ({:.3f} s)'.format(epoch, time.time() - start_time))
#             T_c = epoch % config.T_i
#             eta_min, eta_max = sess.run([vDNN.eta_min, vDNN.eta_max])
#             eta = eta_min + 1/2*(eta_max - eta_min)*(1+np.cos(np.pi*(T_c/config.T_i)))
#             print('Learning rate: {}'.format(eta))
            
            while(readerTrain.proceed):
                # Iterate through entire corpus
                x_train, y_train = readerTrain.next_batch(config.batch_size)
                beta = cosine_anneal(config.learning_rate, global_step, config.T, config.cycles)
                feed_dict_train = {vDNN.x: x_train.values, vDNN.y_true: y_train.values, 
                                   vDNN.keep_prob: config.keep_prob, vDNN.training_phase: True}#, vDNN.beta: beta}
                t_op = sess.run(vDNN.train_op, feed_dict = feed_dict_train)
                step += 1
                global_step += 1

                if step % (config.steps_per_epoch // 5) == 0:            
                    # Evaluate model
                    improved = ''
                    sess.run(tf.local_variables_initializer())
                    x_test, y_test = readerTest.next_batch(config.batch_size)
                    feed_dict_test = {vDNN.x: x_test.values, vDNN.y_true: y_test.values, vDNN.keep_prob: 1.0,
                                      vDNN.training_phase: False}#, vDNN.beta: 0.0}

                    t_acc, t_loss, t_summary = sess.run([vDNN.accuracy, vDNN.cross_entropy, vDNN.merge_op],
                                                        feed_dict = feed_dict_train)
                    v_acc, v_loss, v_auc, v_summary, = sess.run([vDNN.accuracy, vDNN.cross_entropy, vDNN.auc_op, vDNN.merge_op],
                                                        feed_dict = feed_dict_test)

                    vDNN.train_writer.add_summary(t_summary, step)
                    vDNN.test_writer.add_summary(v_summary, step)
                    
                    if epoch > 5 and v_acc > v_acc_best:
                        v_acc_best = v_acc
                        improved = '[*]'
                        save_path = saver.save(sess, 
                                               os.path.join(directories.checkpoints,
                                                            'vDNN_{}_{}_best.ckpt'.format(config.mode, config.channel)),
                                               global_step = epoch)
                    
                    print('Epoch {}, Step {} | Training Acc: {:.3f} | Test Acc: {:.3f} | Test Loss: {:.3f} | Test AUC {:.3f} ({:.2f} s) {}'
                          .format(epoch, step, t_acc, v_acc, v_loss, v_auc, time.time() - start_time, improved))

        save_path = saver.save(sess, os.path.join(directories.checkpoints, 'vDNN_{}_{}_end.ckpt'.format(config.mode, config.channel)),
                               global_step = epoch)
        print('Model saved to file: {}'.format(save_path))

        feed_dict_train = {vDNN.x: df_X_train.values, vDNN.y_true: df_y_train.values, vDNN.keep_prob: 1.0, vDNN.training_phase: False}
        feed_dict_test = {vDNN.x: df_X_test.values, vDNN.y_true: df_y_test.values, vDNN.keep_prob: 1.0, vDNN.training_phase: False}
        final_t_acc = vDNN.accuracy.eval(feed_dict = feed_dict_train)
        final_v_acc, final_v_AUC = sess.run([vDNN.accuracy, vDNN.auc_op], feed_dict = feed_dict_test)

        delta_t = time.time() - start_time
            
    print("Training Complete. Time elapsed: {:.3f} s".format(delta_t))
    print("Train accuracy: {:g}\nValidation accuracy: {:g}\nValidation AUC: {:g}".format(final_t_acc, final_v_acc, final_v_AUC))

    print('Architecture: {}'.format(architecture))
    save_summary(config, delta_t, final_t_acc, final_v_acc, final_v_AUC)


# In[ ]:

train(config)


# In[ ]:

train(config)#, restore = True)


# #### Making Predictions
# Classification on a new instance is given by the softmax of the output of the final readout layer.

# In[12]:

ckpt = tf.train.get_checkpoint_state(directories.checkpoints)
#vDNN = vanillaDNN(config, training = False)
network_output = vDNN.predict(ckpt)

np.save(os.path.join(directories.checkpoints, '{}_{}_y_pred.npy'.format(config.channel, config.mode)), network_output)
np.save(os.path.join(directories.checkpoints, '{}_{}_y_test.npy'.format(config.channel, config.mode)), df_y_test.values)


# In[ ]:



