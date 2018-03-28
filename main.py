# -*- coding: utf-8 -*-
"""
Created on Fri Mar 23 12:30:24 2018

@author: yangyr
"""

import numpy as np
from DataFactory import DataPreprocessor
import train

dp = DataPreprocessor()
X_train,y_train,X_test,y_test = dp.getAllData(dataSize = 'all')

#train.train(X_train,y_train)
