def layer_weights(shape, initializer = tf.contrib.layers.xavier_initializer()):
    # Return weight tensor of given shape using Xavier initialization
    W = tf.get_variable("weights", shape = shape, initializer=initializer)
    return W

def layer_biases(shape, init_value = 0.0):
    # Return bias tensor of given shape with small initialized constant value
    b = tf.get_variable("biases", shape = shape, initializer = tf.constant_initializer(init_value))
    return b

def hidden_SELU_ops(x, shape, name, keep_prob, training=True):
    # Add operations to graph to construct hidden layers
    with tf.variable_scope(name) as scope:
        # scope.reuse_variables() # otherwise tf.get_variable() checks that already existing vars are not shared by accident
        weights = layer_weights(shape = shape, initializer = SELU_initializer)
        biases = layer_biases(shape = [shape[1]])

        # Apply non-linearity. Default is ReLU
        actv = selu.selu(tf.add(tf.matmul(x, weights), biases))
        layer_output = selu.dropout_selu(actv, rate = 1 - keep_prob, training=training)

    return layer_output

def readout_ops(x, shape, name, initializer = tf.contrib.layers.xavier_initializer()):
    # Don't apply non-linearity, dropout on output layer
    with tf.variable_scope(name) as scope:
        weights = layer_weights(shape = shape, initializer = initializer)
        biases = layer_biases(shape = [shape[1]])
        layer_output = tf.matmul(x, weights) + biases

    return layer_output

def BN_layer_ops(x, shape, name, keep_prob, training, activation=tf.nn.relu):
    # High-level implementation of BN
    with tf.variable_scope(name) as scope:
         # scope.reuse_variables() # otherwise tf.get_variable() checks that already existing vars are not shared by accident
        weights = layer_weights(shape = shape)
        biases = layer_biases(shape = [shape[1]])
        z_BN = tf.matmul(x, weights) + biases

        # Place BN transform before non-linearity - update to TF 1.2!
        theta_BN = tf.contrib.layers.batch_norm(z_BN, center=True, scale=True, is_training=training,
                                                decay=0.99, zero_debias_moving_mean=True, scope='bn', fused = True)
        BN_actv = activation(theta_BN)
        BN_layer_output = tf.layers.dropout(BN_actv, keep_prob, training=training)

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
        hidden_1 = builder(x, shape = [config.nFeatures, hidden_layer_nodes[0]], name = 'hidden0',
                                keep_prob = keep_prob, training=training_phase)
        layers.append(hidden_1)
        for n in range(0,n_layers-1):
            hidden_n = builder(layers[-1], shape = [hidden_layer_nodes[n], hidden_layer_nodes[n+1]], name = 'hidden{}'.format(n+1),
                                   keep_prob = keep_prob, training=training_phase)
            layers.append(hidden_n)
        readout = readout_ops(layers[-1], shape = [hidden_layer_nodes[-1], config.n_classes], name = 'readout', initializer = SELU_initializer)

        return readout

def build_SELU_network(x, n_layers, hidden_layer_nodes, keep_prob, training_phase):
    assert n_layers == len(hidden_layer_nodes), 'Specified layer nodes and number of layers do not correspond.'
    layers = [x]
    with tf.variable_scope('SELU_layers') as scope:
        hidden_1 = hidden_SELU_ops(x, shape = [config.nFeatures, hidden_layer_nodes[0]], name = 'SELUhidden0',
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
        hidden_1 = BN_layer_ops(x, shape = [config.nFeatures, hidden_layer_nodes[0]], name = 'BNhidden0',
                                keep_prob = keep_prob, phase = training_phase)
        layers.append(hidden_1)
        for n in range(0,n_layers-1):
            hidden_n = BN_layer_ops(layers[-1], shape = [hidden_layer_nodes[n], hidden_layer_nodes[n+1]], name = 'BNhidden{}'.format(n+1),
                                   keep_prob = keep_prob, phase = training_phase)
            layers.append(hidden_n)
        readout = readout_ops(layers[-1], shape = [hidden_layer_nodes[-1], config.n_classes], name = 'readout')

        return readout
