import numpy as np
import tensorflow as tf


LEARNING_RATE = 0.1       # Learning rate.

def network(inpt):
    """Builds the neural network to approximate Q function

    Args:
        inpt: Input sampled from the stored data.
    Returns:
        out: Tensor representing output layer of the network.
    """
    with tf.variable_scope('fc1') as scope:
        W_1 = tf.placeholder(tf.float32, [16, 48])
        b_1 = tf.placeholder(tf.float32, [48])

        h_1 = tf.matmul(W_1, inpt) + b_1

    with tf.variable_scope('fc2') as scope:
        W_2 = tf.placeholder(tf.float32, [48, 4])
        b_2 = tf.placeholder(tf.float32, [4])

        h_2 = tf.matmul(W_2, h_1) + b_2

    out = tf.nn.softmax(h_2)

def loss(net, labels):
    """Calculates loss for the network.

    Args:
        net: output from network().
        labels: correct labels, 1D tensor of shape [batch_size]
    Returns:
        loss: tensor containing the mean squared error
    """
    loss = tf.reduce_sum(tf.square(labels - net))
    
    # TODO: add loss summary

    return loss
    
def train(loss):
    """Create an optimizer for the Q-Network and apply to variables.

    Args:
        loss: total loss from loss().
    Returns:
        apply_gradient_op: op for training
    """
    opt = tf.train.GradientDescentOptimizer(LEARNING_RATE)
    grads = opt.compute_gradients(total_loss)

    apply_gradient_op = opt.apply_gradients(grads)
    
    # TODO: add summary for gradients 
    
    return apply_gradient_op

