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

def binary_cross_entropy(lable, logits, epsilon = 1e-8):
    logits = tf.clip_by_value(logits, epsilon, 1. - epsilon)
    return tf.reducce_mean(tf.nn.sigmoid_ccross_entropy_with_logits(labels = label, logits = logits))

class gengerator:
    """
        A convolution nerual network with a input of Gauss-Distribution noise.
        The input is a noist with 40x40 pixels.
        The output of this generator is a gray image with 28x28 pixels.
    """
    def __init__(self):
        self.x = tf.placeholder(tf.float32, [None, 1600])
        self.W = tf.Variable(tf.zeros([1600, 784]))
        self.b = tf.Variable(tf.zeros([784]))
        self.y = tf.matmul(x, W) + b
        self.y_ = tf.placeholder(tf.float32, shape = [None, 784])
        self.x_noise = tf.reshape(x, [-1, 40, 40, 1])

        self.W_conv1 = weight_variable([5, 5, 1, 32])
        self.b_conv1 = bias_variable([32])
        self.h_conv1 = tf.nn.relu(conv2d(self.x_noise, self.W_conv1) + self.b_conv1)
        self.h_pool1 = max_pool_2x2(self.h_conv1)

        self.W_conv2 = weight_variable([5, 5, 32, 64])
        self.b_conv2 = bias_variable([64])
        self.h_conv2 = tf.nn.relu(conv2d(self.h_pool1, self.W_conv2) + self.b_conv2)
        self.h_pool2 = max_pool_2x2(self.h_conv2)

        self.W_conv3 = weight_variable([5, 5, 64, 128])
        self.b_conv3 = bias_variable([128])
        self.h_conv3 = tf.nn.relu(conv2d(self.h_pool2, self.W_conv3) + self.b_conv3)
        self.h_pool3 = max_pool_2x2(self.h_conv3)

        self.W_fc1 = weight_variable([7 * 7 * 128, 1024])
        self.b_fc1 = bias_variable([1024])
        self.h_pool3_flat = tf.reshape(self.h_pool3, [-1, 7 * 7 * 128])
        self.h_fc1 = tf.nn.relu(tf.matmul(self.h_pool3_flat, W_fc1) + b_fc1)

        self.keep_prob = tf.placeholder(tf.float32)
        self.h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

        self.W_fc2 = weight_variable([1024, 784])
        self.b_fc2 = bias_variable([784])

        self.y_conv = tf.matmul(self.h_fc1_drop, self.W_fc2) + self.b_fc2


class discriminator:
    """
        The discriminator is another nurual network whose input is an 28x28 image,
        the output is a bool value 0/1, stands for whether the input is follow
        the distribution of MNIST.
    """
    def __init__(self):
        self.w = 1
