# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 22:37:31 2018

@author: Li Zhun

A collection of pseudo-labeling models which will be used to classify unlabeled data.
"""

import sys
import os
from sklearn.externals import joblib

from sklearn import svm
from sklearn import gaussian_process
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel,RBF
from sklearn.ensemble import RandomForestClassifier

class PseudoLabelingModels():
    def __init__(self, nb_classes, model, saved_model=None):
        """
        `model` = one of:svc, gpc, rf(random forest), or deep learning models
        `nb_classes` = the number of classes to predict
        `saved_model` = the path to a saved model to load (joblib.load())
        """

        # Set defaults.
        self.nb_classes = nb_classes
        self.saved_model = saved_model

        # Get the appropriate model.
        if self.saved_model is not None and os.path.isfile(self.saved_model):
            print("Loading model %s" % self.saved_model)
            self.model = joblib.load(self.saved_model)
        elif model == 'svm':
            print("Loading SVM model.")
            self.model = self.svc()
        elif model == 'gpc':
            print("Loading GPC model.")
            self.model = self.gpc()
        elif model == 'rf':
            print("Loading Random Forest.")
            self.model = self.rf()
        else:
            print("Unknown model.")
            sys.exit()

    def svc(self):
        """Create a support vector classifier."""
        #soft margin parameter c
        param_C = 10#5
        #Gaussian kernel parameter gamma
        param_gamma = 0.01#0.017#0.05
        model = svm.SVC(C=param_C,gamma=param_gamma)
        return model

    def gpc(self):
        """Create a Gaussian process classifier"""
        # initiallize kernel
        kernel = ConstantKernel() + Matern(length_scale=2, nu=3/2) + WhiteKernel(noise_level=1)
        # Get the model.
        model = gaussian_process.GaussianProcessClassifier(kernel = kernel, copy_X_train=False)
        return model

    def rf(self):
        """Create a random forest classifier."""
        param_num_trees = 200
        model = RandomForestClassifier(n_estimators = param_num_trees)
        return model

#    def mlp(self):
#        """Build a simple MLP."""
#        # Model.
#        model = Sequential()
#        model.add(Dense(512, input_dim=self.input_shape))
#        model.add(Dropout(0.5))
#        model.add(Dense(512))
#        model.add(Dropout(0.5))
#        model.add(Dense(self.nb_classes, activation='softmax'))
#
#        return model

        
        