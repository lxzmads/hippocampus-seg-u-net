# -*- coding: utf-8 -*-
"""
Created on Tue Mar 20 20:12:05 2018

@author: yangyr
"""

import tensorflow as tf
import tensorlayer as tl
import model

def predict(X_test, y_test):
    ###======================== HYPER-PARAMETERS ============================###
    nw = 256
    nh = 256
    nc = 166
    #print_freq_step = 100
    
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
    
    ###======================== DEFIINE MODEL =======================###
    t_image = tf.placeholder('float32', [1, nw, nh, nc], name='input_image')
    t_seg = tf.placeholder('float32', [1, nw, nh, nc], name='target_segment')
    ## train inference
    net = model.u_net(t_image, is_train=False, reuse=True, n_out=166)
    
    ###======================== DEFINE LOSS =========================###
    out_seg = net.outputs
    dice = 1 - tl.cost.dice_coe(out_seg, t_seg, axis=[0,1,2,3])#, 'jaccard', epsilon=1e-5)
    
    ###======================== LOAD MODEL ==============================###
    tl.layers.initialize_global_variables(sess)
    tl.files.load_and_assign_npz(sess=sess, name='u_net_model.npz', network=net)
    
    ###======================== Predict ================================###
    _dice, out = sess.run([dice, net.outputs],
                    {t_image: X_test, t_seg: y_test})
                
    print("dice: %f" %_dice)
