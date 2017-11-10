#!/usr/bin/env python3
"""
TF Model Playground
"""
import tensorflow as tf
import selu

def dense_model(x, n_layers, hidden_layer_nodes, keep_prob, reuse=False,
        training=True, actv=tf.nn.relu, init=tf.contrib.layers.xavier_initializer()):
    # Sanity check
    with tf.variable_scope('DenseNet', reuse=reuse, initializer=init):
        l0 = tf.layers.dense(x, units=hidden_layer_nodes[0], activation=actv)
        d0 = tf.layers.dropout(l0, keep_prob, training=training)

        l1 = tf.layers.dense(d0, units=hidden_layer_nodes[1], activation=actv)
        d1 = tf.layers.dropout(l1, keep_prob, training=training)

        l2 = tf.layers.dense(d1, units=hidden_layer_nodes[2], activation=actv)
        d2 = tf.layers.dropout(l2, keep_prob, training=training)

        l3 = tf.layers.dense(d2, units=hidden_layer_nodes[3], activation=actv)
        d3 = tf.layers.dropout(l3, keep_prob, training=training)

        l4 = tf.layers.dense(d3, units=hidden_layer_nodes[4], activation=actv)
        d4 = tf.layers.dropout(l4, keep_prob, training=training)

        # Readout layer
        readout = tf.layers.dense(d4, units=config.n_classes, kernel_initializer=init)

    return readout

def dense_BN(x, n_layers, hidden_layer_nodes, keep_prob, reuse=False,
    training=True, actv=tf.nn.relu, init=tf.contrib.layers.xavier_initializer()):
    # Sanity check batch-normed
    kwargs = {'center': True, 'scale': True, 'training': training, 'fused': True, 'renorm': True}

    with tf.variable_scope('BatchNormNet', reuse=reuse, initializer=init):
        l0 = tf.layers.dense(x, units=hidden_layer_nodes[0], activation=actv)
        b0 = tf.layers.batch_normalization(l0, **kwargs)
        d0 = tf.layers.dropout(l0, keep_prob, training=training)

        l1 = tf.layers.dense(d0, units=hidden_layer_nodes[1], activation=actv)
        b1 = tf.layers.batch_normalization(l1, **kwargs)
        d1 = tf.layers.dropout(b1, keep_prob, training=training)

        l2 = tf.layers.dense(d1, units=hidden_layer_nodes[2], activation=actv)
        b2 = tf.layers.batch_normalization(l2, **kwargs)
        d2 = tf.layers.dropout(b2, keep_prob, training=training)

        l3 = tf.layers.dense(d2, units=hidden_layer_nodes[3], activation=actv)
        b3 = tf.layers.batch_normalization(l3, **kwargs)
        d3 = tf.layers.dropout(b3, keep_prob, training=training)

        l4 = tf.layers.dense(d3, units=hidden_layer_nodes[4], activation=actv)
        b4 = tf.layers.batch_normalization(l4, **kwargs)
        d4 = tf.layers.dropout(b4, keep_prob, training=training)

        # Readout layer
        readout = tf.layers.dense(d4, units=config.n_classes, kernel_initializer=init)

    return readout

def dense_SELU(x, n_layers, hidden_layer_nodes, keep_prob, reuse=False,
    training=True, actv=selu.selu):
    SELU_initializer = tf.contrib.layers.variance_scaling_initializer(factor=1.0, mode='FAN_IN')

    with tf.variable_scope('seluNet', reuse=reuse):
        l0 = tf.layers.dense(x, units=hidden_layer_nodes[0], activation=actv,
        kernel_initializer=SELU_initializer)
        d0 = selu.dropout_selu(l0, rate=1-keep_prob, training=training)

        l1 = tf.layers.dense(d0, units=hidden_layer_nodes[1], activation=actv,
        kernel_initializer=SELU_initializer)
        d1 = selu.dropout_selu(l1, rate=1-keep_prob, training=training)

        l2 = tf.layers.dense(d1, units=hidden_layer_nodes[2], activation=actv,
        kernel_initializer=SELU_initializer)
        d2 = selu.dropout_selu(l2, rate=1-keep_prob, training=training)

        l3 = tf.layers.dense(d2, units=hidden_layer_nodes[3], activation=actv,
        kernel_initializer=SELU_initializer)
        d3 = selu.dropout_selu(l3, rate=1-keep_prob, training=training)

        l4 = tf.layers.dense(d3, units=hidden_layer_nodes[4], activation=actv,
        kernel_initializer=SELU_initializer)
        d4 = selu.dropout_selu(l4, rate=1-keep_prob, training=training)

        # Readout layer
        readout = tf.layers.dense(d4, units=config.n_classes,
        kernel_initializer=SELU_initializer)

    return readout

def dense_builder(x, shape, name, keep_prob, training=True, actv=tf.nn.relu):
    init=tf.contrib.layers.xavier_initializer()
    kwargs = {'center': True, 'scale': True, 'training': training, 'fused': True, 'renorm': True}

    with tf.variable_scope(name, initializer=init) as scope:
        layer = tf.layers.dense(x, units=shape[1], activation=actv)
        bn = tf.layers.batch_normalization(layer, **kwargs)
        layer_out = tf.layers.dropout(bn, keep_prob, training=training)

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
