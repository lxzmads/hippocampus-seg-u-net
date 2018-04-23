# -*- coding: utf-8 -*-
"""
Created on Tue Mar 20 17:33:08 2018

@author: yangyr
"""

import numpy as np
import tensorflow as tf
import tensorlayer as tl
import time
import model
import matplotlib.pyplot as plt

def train(X_train, y_train,X_test,y_test):
    ###======================== HYPER-PARAMETERS ============================###
    batch_size = 10
    lr = 0.0001 
    # lr_decay = 0.5
    # decay_every = 100
    beta1 = 0.9
    n_epoch = 45
    
    #print_freq_step = 100
    
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
    
    ###======================== DEFIINE MODEL =======================###
    t_image = tf.placeholder('float32', shape=[batch_size,256,256,1], name='input_image')
    t_seg = tf.placeholder('float32', shape=[batch_size,256,256,1], name='target_segment')
    ## train inference
    net = model.u_net(t_image, is_train=True, reuse=False, n_out=1)
    ## test inference
    net_test = model.u_net(t_image, is_train=False, reuse=True, n_out=1)
    
    ###======================== DEFINE LOSS =========================###
    ## train losses
    out_seg = net.outputs
    dice_loss = 1 - tl.cost.dice_coe(out_seg, t_seg, axis=[0,1,2,3])#, 'jaccard', epsilon=1e-5)
    loss = dice_loss
    
    ## test losses
    test_out_seg = net_test.outputs
    test_dice_loss = 1 - tl.cost.dice_coe(test_out_seg, t_seg, axis=[0,1,2,3])#, 'jaccard', epsilon=1e-5)
    
    ###======================== DEFINE TRAIN OPTS =======================###
    t_vars = tl.layers.get_variables_with_name('u_net', True, True)
    with tf.variable_scope('learning_rate'):
        lr_v = tf.Variable(lr, trainable=False)
    train_op = tf.train.AdamOptimizer(lr_v, beta1=beta1).minimize(loss, var_list=t_vars)
    
    ###======================== LOAD MODEL ==============================###
    tl.layers.initialize_global_variables(sess)
    
    ###======================== TRAINING ================================###
    dices = []
    for epoch in range(0, n_epoch+1):
        epoch_time = time.time()
        total_dice, n_batch = 0, 0
        #for image, label in X_train, y_train:
            
        for batch in tl.iterate.minibatches(inputs=X_train, targets=y_train,
                                   batch_size=batch_size, shuffle=True):
            images, labels = batch
            #step_time = time.time()

            ## update network
            _, _dice, out = sess.run([train_op,
                    loss, net.outputs],
                    {t_image: images, t_seg: labels})
            _dice = 1 - _dice
            total_dice += _dice
            n_batch += 1
            
            ## check model fail
            if np.isnan(_dice):
                exit(" ** NaN loss found during training, stop training")
            if np.isnan(out).any():
                exit(" ** NaN found in output images during training, stop training")
        
        dices.append(total_dice/n_batch)
        print(" ** Epoch [%d/%d] train dice: %f took %fs (2d with distortion)" %
              (epoch, n_epoch, total_dice/n_batch, time.time()-epoch_time))  
        
    tl.files.save_npz(net.all_params, name='u_net_model.npz', sess=sess)
    plt.figure()
    plt.plot(range(0, n_epoch+1),dices)
    plt.show()
    
    total_dice_test, n_batch = 0, 0
    epoch_time = time.time()
    for batch in tl.iterate.minibatches(inputs=X_test, targets=y_test,
                                   batch_size=batch_size, shuffle=False):
        images, labels = batch
        #step_time = time.time()

        ## update network
        _dice_test, out_test = sess.run([test_dice_loss, net_test.outputs],
                {t_image: images, t_seg: labels})
        _dice_test = 1 - _dice_test
        total_dice_test += _dice_test
        n_batch += 1
        
        ## check model fail
        if np.isnan(_dice_test):
            exit(" ** NaN loss found during training, stop training")
        if np.isnan(out_test).any():
            exit(" ** NaN found in output images during training, stop training")

    print(" ** Epoch [%d/%d] test dice: %f took %fs (2d with distortion)" %
          (epoch, n_epoch, total_dice_test/n_batch, time.time()-epoch_time))  
