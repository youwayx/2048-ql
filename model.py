import numpy as np
import tensorflow as tf


LEARNING_RATE = 0.001       # Learning rate.

def network(inpt):
    """Builds the neural network to approximate Q function

    Args:
        inpt: Input sampled from the stored data.
    Returns:
        out: Tensor representing output layer of the network.
    """
    with tf.variable_scope('fc1') as scope:
        W_1 = tf.Variable(tf.truncated_normal([16, 48], stddev=0.1))
        b_1 = tf.Variable(tf.zeros( [48]))

        h_1 = tf.nn.relu(tf.matmul(inpt, W_1) + b_1)

    with tf.variable_scope('fc2') as scope:
        W_2 = tf.Variable(tf.truncated_normal([48, 4], stddev=0.1))
        b_2 = tf.Variable(tf.zeros([4]))

        out = tf.nn.relu(tf.matmul(h_1, W_2) + b_2)

    return out

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
    opt = tf.train.AdamOptimizer(LEARNING_RATE)
    grads = opt.compute_gradients(loss)

    apply_gradient_op = opt.apply_gradients(grads)
    
    # TODO: add summary for gradients 
    
    return apply_gradient_op

def feed_forward(net, feed_dict, session):
    return net.eval(feed_dict=feed_dict, session=session)
