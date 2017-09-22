# -*- coding: utf-8 -*-
# Functions for multi-gpu training
import tensorflow as tf

def tower_computation(model, config, scope, examples, labels, training, n_gpu):
    """ Calculate the total loss on a single computation tower.
    Args:
        model: function to build inference graph
        scope: unique prefix string identifying the tower, e.g. 'tower_0'
        examples: 2D tensor of shape [batch_size, n_features]
        labels: 1D tensor of shape [batch_size]
    Returns:
        cross_entropy: Tensor containing the total loss for a batch of data
        readout: logits of readout layer
    """
    # Build inference graph
    readout = model(examples, config.n_layers, config.hidden_layer_nodes, config.keep_prob, training=training)

    # Get losses - try L2 loss?
    cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=readout, labels=labels))
    tf.add_to_collection('losses_collection', cross_entropy)

    # Assemble all of the losses for the current tower only.
    losses = tf.get_collection('losses_collection', scope)

    for l in losses:
        tf.summary.scalar('xentropy_{}-raw'.format(n_gpu), l)

    return cross_entropy, readout

def average_gradients(tower_grads):
    """ Calculate the average gradient for each shared variable across all towers.
    Args:
    tower_grads: Nested list of (gradient, variable) tuples. The outer list
      is over individual gradients. The inner list is over the gradient
      calculation for each tower.
    Returns:
     List of pairs of (gradient, variable) where the gradient has been averaged
     across all towers.
    """
    average_grads = []
    for grad_var_pair in zip(*tower_grads):
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = []
        for g, _ in grad_var_pair:
            # Add 0 dimension to the gradients to represent the tower.
            expanded_g = tf.expand_dims(g, 0)
            # Append on a 'tower' dimension which we will average over below.
            grads.append(expanded_g)
        # Average over the 'tower' dimension.
        grad = tf.concat(axis=0, values=grads)
        grad = tf.reduce_mean(grad, 0)

        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So just return the first tower's pointer to
        # the Variable.
        v = grad_var_pair[0][1]
        gv_pair = (grad, v)
        average_grads.append(gv_pair)

    return average_grads
