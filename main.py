# -*- coding: utf-8 -*-
"""
Created on Fri Mar 23 12:30:24 2018

@author: yangyr
"""

import numpy as np
from DataFactory import DataPreprocessor
import train

dp = DataPreprocessor()
X_train,y_train,X_test,y_test = dp.getAllData(dataSize = 'small')
X_train = X_train[:,:,:,np.newaxis]
y_train = y_train[:,:,:,np.newaxis]
X_test = X_test[:,:,:,np.newaxis]
y_test = y_test[:,:,:,np.newaxis]

train.train(X_train,y_train,X_test,y_test)
