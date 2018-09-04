#!/bin/env python3
#-*-coding=utf-8-*-
#define CNN architecture

import os, sys
import numpy as np
import tensorflow as tf


class CNNmodel(object):
    def __init__(self):
        
        return

    def weight_variable(self, shape, name):
        return tf.get_variable(shape=shape,dtype=tf.float32, name=name, initializer=tf.truncated_normal_initializer(stddev=0.7))

    def bias_variable(self, shape, name):
        return tf.get_variable(shape=shape,dtype=tf.float32, name=name, initializer=tf.constant_initializer(0.0))

    def conv2dV(self, x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='VALID')

    def conv2dS(self, x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
    
    def max_pool_2x2(self, x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                              strides=[1, 2, 2, 1], padding='VALID')
    def max_pool_3x3(self, x):
        return tf.nn.max_pool(x, ksize=[1, 3, 3, 1],
                              strides=[1, 3, 3, 1], padding='VALID')
    
    def max_pool_1x3(self,x):
        return tf.nn.max_pool(x, ksize=[1, 3, 3, 1],
                              strides=[1, 1, 3, 1], padding='VALID')
    
    def max_pool_3x1(self, x):
        return tf.nn.max_pool(x, ksize=[1, 3, 3, 1],
                              strides=[1, 3, 1, 1], padding='VALID')
    def max_pool_3x2(self, x):
        return tf.nn.max_pool(x, ksize=[1, 3, 3, 1],
                              strides=[1, 3, 2, 1], padding='VALID')
    
    
    def max_pool_3x3_2(self, x):
        return tf.nn.max_pool(x, ksize=[1, 3, 3, 1],
                              strides=[1, 2, 2, 1], padding='VALID')
    
    def max_pool_5x5(self, x):
        return tf.nn.max_pool(x, ksize=[1, 5, 5, 1],
                              strides=[1, 3, 3, 1], padding='VALID')

    def norm(self, x_image, is_training):
        """                                                                
        parameter:                                                     
        x_image: image input
        is_training: flag to indicate this batch norm layer applying to training or testing
        scale: When the next layer is linear (also e.g. nn.relu), this can be disabled since the scaling can be done by the next layer
        """
        x_out = tf.layers.batch_normalization(x_image, momentum=0.99, scale=False, training=is_training)
        return x_out
    
    def init_weight(self):
        #FirstLayer Convolution 1
        self.W_conv1_1 = self.weight_variable([1, 1, 1, 8], name='wconv1_1')
        self.b_conv1_1 = self.bias_variable([8], name='bconv1_1')

        #SecondLayer Convolution 1
        self.W_conv2_1 = self.weight_variable([3, 3, 8, 8], name='wconv2_1')
        self.b_conv2_1 = self.bias_variable([8], name='bconv2_1')

        #FirstLayer Convolution 2
        self.W_conv1_2 = self.weight_variable([1, 1, 1, 8], name='wconv1_2')
        self.b_conv1_2 = self.bias_variable([8], name='bconv1_2')

        #SecondLayer Convolution 2
        self.W_conv2_2 = self.weight_variable([5, 5, 8, 8], name='wconv2_2')
        self.b_conv2_2 = self.bias_variable([8], name='bconv2_2')

        #FirstLayer Convolution 3
        self.W_conv1_3 = self.weight_variable([1, 1, 1, 8], name='wconv1_3')
        self.b_conv1_3 = self.bias_variable([8], name='bconv1_3')

        #SecondLayer Convolution 3
        self.W_conv2_3 = self.weight_variable([7, 7, 8, 8], name='wconv2_3')
        self.b_conv2_3 = self.bias_variable([8], name='bconv2_3')


        #FirstLayer Convolution Final
        self.W_conv1_f = self.weight_variable([3, 3, 24, 24], name='wconv1_f')
        self.b_conv1_f = self.bias_variable([24], name='bconv1_f')

        #SecondLayer Convolution Final
        self.W_conv2_f = self.weight_variable([3, 3, 24, 28], name='wconv2_f')
        self.b_conv2_f = self.bias_variable([28], name='bconv2_f')

        #FullConnect 1
        self.W_fc1 = self.weight_variable([700, 320], name='wfc1')
        self.b_fc1 = self.bias_variable([320], name='bfc1')

        #FullyConnect 2
        self.W_fc2 = self.weight_variable([320, 64], name='wfc2')
        self.b_fc2 = self.bias_variable([64], name='bfc2')


        #FullyConnect 3
        self.W_fc3 = self.weight_variable([64, 1], name='wfc3')
        self.b_fc3 = self.bias_variable([1], name='bfc3')

    def model(self, input_images, sess, keep_prob, reuse_weight=False, is_training=True):
        """
        Build the flowing chart here
        The structure include one layer inception (1X1 3X3), (1X1 5X5)
        """
        if (reuse_weight==False):
            self.init_weight()

        with tf.name_scope("cnn_inception"):
            # normalize the crystal arrays first
            x_image_norm = self.norm(input_images, is_training)
        
            #FirstLayer Convolution 1
            h_conv1_1 = tf.nn.relu(self.conv2dS(x_image_norm, self.W_conv1_1) + self.b_conv1_1)
            first_conv_shape_1= tf.shape(h_conv1_1)
            print(sess.run(first_conv_shape_1))
        
            #SecondLayer Convolution 1
            h_conv2_1 = tf.nn.relu(self.norm(self.conv2dS(h_conv1_1, self.W_conv2_1) + self.b_conv2_1, is_training))
            h_pool2_1 = self.max_pool_3x3(h_conv2_1)
            second_conv_shape_1= tf.shape(h_pool2_1)
            print(sess.run(second_conv_shape_1))

            #FirstLayer Convolution 2
            h_conv1_2 = tf.nn.relu(self.conv2dS(x_image_norm, self.W_conv1_2) + self.b_conv1_2)
            first_conv_shape_2= tf.shape(h_conv1_2)
            print(sess.run(first_conv_shape_2))

            #SecondLayer Convolution 2
            h_conv2_2 = tf.nn.relu(self.norm(self.conv2dS(h_conv1_2, self.W_conv2_2) + self.b_conv2_2, is_training))
            h_pool2_2 = self.max_pool_3x3(h_conv2_2)
            second_conv_shape_2= tf.shape(h_pool2_2)
            print(sess.run(second_conv_shape_2))

            #FirstLayer Convolution 3
            h_conv1_3 = tf.nn.relu(self.conv2dS(x_image_norm, self.W_conv1_3) + self.b_conv1_3)
            first_conv_shape_3= tf.shape(h_conv1_3)
            print(sess.run(first_conv_shape_3))

            #SecondLayer Convolution 3
            h_conv2_3 = tf.nn.relu(self.norm(self.conv2dS(h_conv1_3, self.W_conv2_3) + self.b_conv2_3, is_training))
            h_pool2_3 = self.max_pool_3x3(h_conv2_3)
            second_conv_shape_3= tf.shape(h_pool2_3)
            print(sess.run(second_conv_shape_3))

            
            h_concat_all = tf.concat([h_pool2_1, h_pool2_2, h_pool2_3], 3)
            h_concat_all_shape= tf.shape(h_concat_all)
            print(sess.run(h_concat_all_shape))
        
            #FirstLayer Convolution Final
            h_conv1_f = tf.nn.relu(self.norm(self.conv2dS(h_concat_all, self.W_conv1_f) + self.b_conv1_f, is_training))
            #h_pool1_f = self.max_pool_2x2(h_conv1_f)
            #first_conv_shape_f= tf.shape(h_pool1_f)
            #print(sess.run(first_conv_shape_f))

            #SecondLayer Convolution Final
            #h_conv2_f = tf.nn.relu(self.norm(self.conv2dS(h_pool1_f, self.W_conv2_f) + self.b_conv2_f, is_training))
            h_conv2_f = tf.nn.relu(self.norm(self.conv2dS(h_conv1_f, self.W_conv2_f) + self.b_conv2_f, is_training))
            h_pool2_f = self.max_pool_2x2(h_conv2_f)

            final_conv_shape = list(sess.run(tf.shape(h_pool2_f)))
            print("Final conv shape {}".format(final_conv_shape))

            final_shape = final_conv_shape[1]*final_conv_shape[2]*final_conv_shape[3]

            #Flatten
            h_pool2_f_flat = tf.reshape(h_pool2_f, [final_conv_shape[0], final_shape])

            #drop out keep number
            #keep_prob = tf.placeholder(tf.float32)
            #Fully connected first layer (drop out)
            h_fc1 = tf.nn.relu(self.norm(tf.matmul(h_pool2_f_flat, self.W_fc1) + self.b_fc1, is_training))
            h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

            #Fully connect second layer (drop out)
            h_fc2 = tf.nn.relu(self.norm(tf.matmul(h_fc1_drop, self.W_fc2) + self.b_fc2, is_training))
            h_fc2_drop = tf.nn.dropout(h_fc2, keep_prob)

            #Fully connected second layer 
            y_out = tf.nn.sigmoid(tf.matmul(h_fc2_drop, self.W_fc3) + self.b_fc3, name="output")

        return y_out

    def loss(self, y_predict, y_true):
        """
        Loss function: MSE
        """
        sum_of_square = tf.reduce_mean(tf.reduce_sum(tf.square(y_true - y_predict), reduction_indices=[1]), name="mse")
        abs_error =  tf.reduce_mean(tf.reduce_sum(tf.abs(y_true - y_predict), reduction_indices=[1]), name="abserr")
        mse = sum_of_square
        error = abs_error

        return (mse, error)

    def optimizer_step1(self, mse_loss, learn_rate=1e-3):
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_step = tf.train.AdamOptimizer(learn_rate).minimize(mse_loss)

        return train_step
    
    def optimizer_step2(self, mse_loss, learn_rate=1e-4):
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_step_2 = tf.train.AdamOptimizer(learn_rate).minimize(mse_loss)

        return train_step_2

    """
    def result(self, y_predict):
        #y_result = tf.reduce_mean(tf.reduce_sum(y_predict, reduction_indices=[1]))
        y_result = tf.squeeze(y_predict)
        return y_result
    """
