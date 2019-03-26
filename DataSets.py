# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 21:39:58 2018

@author: Li Zhun

A collection of data sets with necessary preprocessing.
Original data set will be diveded into six parts, i.e. 
(x_train, y_train), (x_test, y_test), (x_unlabel, y_unlabel)
"""

import sys
import numpy as np
import time

from keras.datasets import mnist, cifar10, cifar100, imdb
from keras.preprocessing import sequence


class DataSets():
    def __init__(self, data_name, sample_per_class = None):
        """
        `data_name` = one of: mlp_mnist, cnn_mnist, cnn_cifar10, lstm_imdb, lstm_ucf101
        `sample_per_class` = the sampling number of each class, None means all.
        """
        # Set defaults.
        self.sample_per_class = sample_per_class
        
        # Get the appropriate data set.
        if data_name == 'mlp_mnist':
            print("Loading MNIST data set.")
            self.data = self.load_mnist_mlp()
        elif data_name == 'cnn_mnist':
            print("Loading MNIST data set.")
            self.data = self.load_mnist_cnn()
        elif data_name == 'cnn_cifar10':
            print("Loading CIFAR10 data set.")
            self.data = self.load_cifar10()
        elif data_name == 'cnn_cifar100':
            print("Loading CIFAR100 data set.")
            self.data = self.load_cifar100()
        elif data_name == 'lstm_imdb':
            print("Loading imdb data set.")
            self.data = self.load_imdb()
        elif data_name == 'lstm_ucf101':
            print("Loading ucf101 data set.")
            self.data = self.load_ucf101()
        else:
            print("Unknown data set.")
            sys.exit()
            
        #Sampling  
        (x_train, y_train), (x_test, y_test) = self.data
        sample_index_train=self.random_sampling_by_class(self.sample_per_class, y_train)
        x_tr = x_train[sample_index_train]
        y_tr = y_train[sample_index_train]
        #compute the complementary set of sample_index
        sample_index_unl = list(set(np.arange(len(y_train))).difference(set(sample_index_train)))
        x_unlabel = x_train[sample_index_unl]
        y_unlabel = y_train[sample_index_unl]
        print('Total: %d train samples, %d test samples' % (x_train.shape[0], x_test.shape[0]))
        print('Use: %d train samples, %d test samples, %d unlabeled samples' 
              % (x_tr.shape[0], x_test.shape[0], x_unlabel.shape[0]))
        
        self.data = (x_tr, y_tr), (x_test, y_test), (x_unlabel, y_unlabel)
        
    def load_mnist_mlp(self):
        """Load MNIST data set. Do necessary preprocessing.
        Starting version from:
        https://github.com/keras-team/keras/blob/master/examples/mnist_mlp.py
        """
        # the data, shuffled and split between train and test sets
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_train = x_train.reshape(60000, 784)
        x_test = x_test.reshape(10000, 784)
        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        x_train /= 255
        x_test /= 255
        return (x_train, y_train), (x_test, y_test)

    def load_mnist_cnn(self):
        """Load MNIST data set. Do necessary preprocessing.
        Starting version from:
        https://github.com/keras-team/keras/blob/master/examples/mnist_mlp.py
        """
        # the data, shuffled and split between train and test sets
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_train = x_train.reshape(60000, 28,28,1)
        x_test = x_test.reshape(10000, 28,28,1)
        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        x_train /= 255
        x_test /= 255
        return (x_train, y_train), (x_test, y_test)

    def load_cifar10(self):
        """Build a CNN for CIFAR10 data set.
        Starting version from:
        https://github.com/keras-team/keras/blob/master/examples/cifar10_cnn.py
        """
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        y_train = y_train.ravel()
        y_test = y_test.ravel()
        return (x_train, y_train), (x_test, y_test)

    def load_cifar100(self):
        """Build a CNN for CIFAR100 data set.
        Starting version from:
        https://keras.io/datasets/
        """
        (x_train, y_train), (x_test, y_test) = cifar100.load_data(label_mode='fine')
        y_train = y_train.ravel()
        y_test = y_test.ravel()
        return (x_train, y_train), (x_test, y_test)

    def load_imdb(self):
        """Build a LSTM network for imdb data set."""
        max_features = 20000
        maxlen = 80  # cut texts after this number of words (among top max_features most common words)
        (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features, seed=int(time.time()))      
        x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
        x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
        return (x_train, y_train), (x_test, y_test)

    def load_ucf101(self):
        """Build a simple LSTM network. We pass the extracted features from
        CNN pre-trained on imagenet to this model predomenently.
        Starting version from:
        https://github.com/harvitronix/five-video-classification-methods
        """
        return 1
    
    @staticmethod    
    def random_sampling_by_class(n_sample, y):
        """Choose n_sample samples from each class. 
        y is training data label. 
        Return sample index.
        """
        sample_index=[]
        for i in range(np.max(y)+1):
            class_size=len(y[y==i])
            class_index=np.arange(len(y))[y==i]
            if class_size<n_sample:
                sample_index.append(class_index)
                sample_index.append(  np.random.choice(class_index,n_sample-class_size,replace=True)    )
            if class_size>=n_sample:
                sample_index.append(  np.random.choice(class_index,n_sample,replace=False)    )
        return np.concatenate(sample_index)