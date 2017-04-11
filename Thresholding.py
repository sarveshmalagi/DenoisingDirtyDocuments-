#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  9 14:49:14 2017

@author: sarvesh
"""
import cv2
import os
import xgboost as xgb
import numpy as np
from sklearn.externals import joblib

thresholdedIntensities = np.empty([0,1])

#The directory has the training image set
#iterate for every file in the directory
for filename in os.listdir('train/'):

    dirtyImg = cv2.imread(os.path.join('train',filename),cv2.IMREAD_GRAYSCALE)
    ret,dirtyImg = cv2.threshold(dirtyImg,30,255,cv2.THRESH_BINARY)
    #TBD : find the threshold using KMeans clustering
    dirtyImg = dirtyImg.reshape(np.product(np.shape(dirtyImg)),1)
    thresholdedIntensities = np.vstack((thresholdedIntensities,dirtyImg))

joblib.dump(thresholdedIntensities,'threshold.model')
