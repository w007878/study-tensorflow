"""
    This code is a Generative Adversial Network to generate a MNIST digit.
    The main model are two convolution nerual networks. Be modified from
    tensorflow tutorial.
"""

import tensorflow as tf
import numpy as np
import os
import cv2
import argparse

from tensorflow.examples.tutorials.mnist import input_data

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev = 0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape = shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides = [1, 1, 1, 1], padding = 'SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize = [1, 2, 2, 1],
                         strides = [1, 2, 2, 1], padding = 'SAME')
class gengerator:
    """
        A convolution nerual network with a input of Gauss-Distribution noise.
        The output of this generator is a gray image with 28x28 pixels.
    """
    def __init__():
        self.a = 1

class discriminator:
    """
        The discriminator is another nurual network whose input is an 28x28 image,
        the output is a bool value 0/1, stands for whether the input is follow
        the distribution of MNIST.
    """
    def __init__():
        w
