# -*- coding: utf-8 -*-
"""
Created on Wed Apr 18 23:05:46 2018

@author: yangyr
"""

import tensorflow as tf
import tensorlayer as tl
import nibabel as nib
import model
import numpy as np

def predict():
    imgPath = "C:/Users/yangyr/Desktop/hippocampus-seg-u-net/images/ADNI_002_S_1018_MR_MPR__GradWarp__B1_Correction__N3__Scaled_Br_20070217030439623_S23128_I40817.nii.gz"
    img = nib.load(imgPath)
    imgData = img.get_data() #(256, 256, 166, 1)
    # get the first three dimension - 3D data
    imgData = imgData[:,:,:,0]
    imgData = imgData[:,:,:,np.newaxis]
    ###======================== HYPER-PARAMETERS ============================###
    bs = imgData.shape[0]
    nw = imgData.shape[1]
    nh = imgData.shape[2]
    nc = imgData.shape[3]
    #print_freq_step = 100
    
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
    
    ###======================== DEFIINE MODEL =======================###
    t_image = tf.placeholder('float32', [bs, nw, nh, nc], name='input_image')
    ## train inference
    net = model.u_net(t_image, is_train=False, reuse=True, n_out=1)
    
    ###======================== LOAD MODEL ==============================###
    tl.layers.initialize_global_variables(sess)
    tl.files.load_and_assign_npz(sess=sess, name='u_net_model.npz', network=net)
    
    ###======================== Predict ================================###
    out = sess.run([net.outputs],
                    {t_image: imgData})
    nib.save(out,"predict.nii")
