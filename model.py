# -*- coding: utf-8 -*-
"""
Created on Tue Mar 20 16:15:16 2018

@author: yangyr
"""

import tensorflow as tf
import tensorlayer as tl
import numpy as np

def u_net(x, is_train=False, reuse=False, n_out=1):
    _, nx, ny, nz = x.get_shape().as_list()
    with tf.variable_scope("u_net", reuse=reuse):
        tl.layers.set_name_reuse(reuse)
        inputs = tl.layers.InputLayer(x, name='inputs')
        conv1 = tl.layers.Conv2d(inputs, 64, (3, 3), act=tf.nn.relu, name='conv1_1')
        conv1 = tl.layers.Conv2d(conv1, 64, (3, 3), act=tf.nn.relu, name='conv1_2')
        pool1 = tl.layers.MaxPool2d(conv1, (2, 2), name='pool1')
        conv2 = tl.layers.Conv2d(pool1, 128, (3, 3), act=tf.nn.relu, name='conv2_1')
        conv2 = tl.layers.Conv2d(conv2, 128, (3, 3), act=tf.nn.relu, name='conv2_2')
        pool2 = tl.layers.MaxPool2d(conv2, (2, 2), name='pool2')
        conv3 = tl.layers.Conv2d(pool2, 256, (3, 3), act=tf.nn.relu, name='conv3_1')
        conv3 = tl.layers.Conv2d(conv3, 256, (3, 3), act=tf.nn.relu, name='conv3_2')
        pool3 = tl.layers.MaxPool2d(conv3, (2, 2), name='pool3')
        conv4 = tl.layers.Conv2d(pool3, 512, (3, 3), act=tf.nn.relu, name='conv4_1')
        conv4 = tl.layers.Conv2d(conv4, 512, (3, 3), act=tf.nn.relu, name='conv4_2')
        pool4 = tl.layers.MaxPool2d(conv4, (2, 2), name='pool4')
        conv5 = tl.layers.Conv2d(pool4, 1024, (3, 3), act=tf.nn.relu, name='conv5_1')
        conv5 = tl.layers.Conv2d(conv5, 1024, (3, 3), act=tf.nn.relu, name='conv5_2')

        up4 = tl.layers.DeConv2d(conv5, 512, (3, 3), (nx/8, ny/8), (2, 2), name='deconv4')
        up4 = tl.layers.ConcatLayer([up4, conv4], 3, name='concat4')
        conv4 = tl.layers.Conv2d(up4, 512, (3, 3), act=tf.nn.relu, name='uconv4_1')
        conv4 = tl.layers.Conv2d(conv4, 512, (3, 3), act=tf.nn.relu, name='uconv4_2')
        up3 = tl.layers.DeConv2d(conv4, 256, (3, 3), (nx/4, ny/4), (2, 2), name='deconv3')
        up3 = tl.layers.ConcatLayer([up3, conv3], 3, name='concat3')
        conv3 = tl.layers.Conv2d(up3, 256, (3, 3), act=tf.nn.relu, name='uconv3_1')
        conv3 = tl.layers.Conv2d(conv3, 256, (3, 3), act=tf.nn.relu, name='uconv3_2')
        up2 = tl.layers.DeConv2d(conv3, 128, (3, 3), (nx/2, ny/2), (2, 2), name='deconv2')
        up2 = tl.layers.ConcatLayer([up2, conv2], 3, name='concat2')
        conv2 = tl.layers.Conv2d(up2, 128, (3, 3), act=tf.nn.relu,  name='uconv2_1')
        conv2 = tl.layers.Conv2d(conv2, 128, (3, 3), act=tf.nn.relu, name='uconv2_2')
        up1 = tl.layers.DeConv2d(conv2, 64, (3, 3), (nx/1, ny/1), (2, 2), name='deconv1')
        up1 = tl.layers.ConcatLayer([up1, conv1] , 3, name='concat1')
        conv1 = tl.layers.Conv2d(up1, 64, (3, 3), act=tf.nn.relu, name='uconv1_1')
        conv1 = tl.layers.Conv2d(conv1, 64, (3, 3), act=tf.nn.relu, name='uconv1_2')
        conv1 = tl.layers.Conv2d(conv1, n_out, (1, 1), act=tf.nn.sigmoid, name='uconv1')
    return conv1