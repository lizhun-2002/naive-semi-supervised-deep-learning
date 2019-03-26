# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 22:44:03 2018

@author: Li Zhun

A collection of Deep-Learning models.
Each of them will be used to classify a specified data set, such as MNIST, CIFAR10 etc.
"""

import sys
import os

from keras.layers import Dense, Flatten, Dropout, Activation
from keras.models import Sequential, load_model
from keras.optimizers import Adam, RMSprop, Adadelta
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.layers import LSTM, Embedding

class DeepLearningModels():
    def __init__(self, nb_classes, model, saved_model=None):
        """
        `model` = one of: mlp_mnist, cnn_mnist, cnn_cifar10, lstm_imdb, lstm_ucf101
        `nb_classes` = the number of classes to predict
        `saved_model` = the path to a saved Keras model to load
        """

        # Set defaults.
        self.load_model = load_model
        self.saved_model = saved_model
        self.nb_classes = nb_classes
        
        # Get the appropriate model.
        if self.saved_model is not None and os.path.isfile(self.saved_model):
            print("Loading model %s" % self.saved_model)
            self.model = load_model(self.saved_model)
        elif model == 'mlp_mnist':
            print("Loading MLP model for MNIST data set.")
            self.model = self.mlp_mnist()
        elif model == 'cnn_mnist':
            print("Loading CNN model for MNIST data set.")
            self.model = self.cnn_mnist()
        elif model == 'cnn_cifar10':
            print("Loading CNN model for CIFAR10 data set.")
            self.model = self.cnn_cifar10()
        elif model == 'cnn_cifar100':
            print("Loading CNN model for CIFAR100 data set.")
            self.model = self.cnn_cifar100()
        elif model == 'lstm_imdb':
            print("Loading LSTM model for imdb data set.")
            self.model = self.lstm_imdb()
        elif model == 'lstm_ucf101':
            print("Loading LSTM model for ucf101 data set.")
            self.model = self.lstm_ucf101()
        else:
            print("Unknown network.")
            sys.exit()


    def mlp_mnist(self):
        """Build a simple MLP for MNIST data set.
        Starting version from:
        https://github.com/keras-team/keras/blob/master/examples/mnist_mlp.py
        """
        # Model.
        model = Sequential()
        model.add(Dense(512, activation='relu', input_shape=(784,)))
        model.add(Dropout(0.2))
        model.add(Dense(512, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(self.nb_classes, activation='softmax'))
#        model.summary()
        model.compile(loss='categorical_crossentropy',
                      optimizer=RMSprop(),
                      metrics=['accuracy', 'top_k_categorical_accuracy'])
        return model

    def cnn_mnist(self):
        """Build a simple CNN for MNIST data set.
        Starting version from:
        https://github.com/keras-team/keras/blob/master/examples/mnist_cnn.py
        """
        # Define the model.
        model = Sequential()
        model.add(Conv2D(32, kernel_size=(3, 3),
                         activation='relu',
                         input_shape=(28, 28, 1)))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(self.nb_classes, activation='softmax'))
        
        model.compile(loss='categorical_crossentropy',
                      optimizer=Adadelta(),
                      metrics=['accuracy', 'top_k_categorical_accuracy'])
        return model

    def cnn_cifar10(self):
        """Build a CNN for CIFAR10 data set.
        Starting version from:
        https://github.com/keras-team/keras/blob/master/examples/cifar10_cnn.py
        """
        model = Sequential()
        model.add(Conv2D(32, (3, 3), padding='same',
                         input_shape=(32, 32, 3)))
        model.add(Activation('relu'))
        model.add(Conv2D(32, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        
        model.add(Conv2D(64, (3, 3), padding='same'))
        model.add(Activation('relu'))
        model.add(Conv2D(64, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        
        model.add(Flatten())
        model.add(Dense(512))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        model.add(Dense(self.nb_classes))
        model.add(Activation('softmax'))
        
        # initiate RMSprop optimizer
        opt = RMSprop(lr=0.0001, decay=1e-6)
        
        # Let's train the model using RMSprop
        model.compile(loss='categorical_crossentropy',
                      optimizer=opt,
                      metrics=['accuracy', 'top_k_categorical_accuracy'])
        return model

    def cnn_cifar10_big(self):
        """Build a CNN for CIFAR10 data set.
        Starting version from:
        https://github.com/keras-team/keras/blob/master/examples/cifar10_cnn.py
        """
        model = Sequential()
        model.add(Conv2D(96, (3, 3), activation='relu', padding='same',
                         input_shape=(32, 32, 3)))
        model.add(Conv2D(96, (3, 3), activation='relu'))
        model.add(Conv2D(96, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        
        model.add(Conv2D(192, (3, 3), activation='relu', padding='same'))
        model.add(Conv2D(192, (3, 3), activation='relu'))
        model.add(Conv2D(192, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        
        model.add(Conv2D(192, (3, 3), activation='relu'))
        model.add(Conv2D(192, (1, 1), activation='relu'))
        model.add(Conv2D(10, (1, 1), activation='relu'))
        model.add(GlobalAveragePooling2D())
        model.add(Activation('softmax'))

        # initiate RMSprop optimizer
        opt = RMSprop(lr=0.0001, decay=1e-6)
        
        # Let's train the model using RMSprop
        model.compile(loss='categorical_crossentropy',
                      optimizer=opt,
                      metrics=['accuracy', 'top_k_categorical_accuracy'])
        return model

    def cnn_cifar100(self):
        """Build a CNN for CIFAR10 data set.
        Starting version from:
        https://github.com/keras-team/keras/blob/master/examples/cifar10_cnn.py
        """
        model = Sequential()
        model.add(Conv2D(32, (3, 3), padding='same',
                         input_shape=(32, 32, 3)))
        model.add(Activation('relu'))
        model.add(Conv2D(32, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        
        model.add(Conv2D(64, (3, 3), padding='same'))
        model.add(Activation('relu'))
        model.add(Conv2D(64, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        
        model.add(Flatten())
        model.add(Dense(512))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        model.add(Dense(self.nb_classes))
        model.add(Activation('softmax'))
        
        # initiate RMSprop optimizer
        opt = RMSprop(lr=0.0001, decay=1e-6)
        
        # Let's train the model using RMSprop
        model.compile(loss='categorical_crossentropy',
                      optimizer=opt,
                      metrics=['accuracy', 'top_k_categorical_accuracy'])
        return model

    def lstm_imdb(self):
        """Build a LSTM network for imdb data set."""
        max_features = 20000
        # Model.
        model = Sequential()
        model.add(Embedding(max_features, 128))
        model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
#        model.add(Dense(1, activation='sigmoid'))
        model.add(Dense(self.nb_classes, activation='softmax'))
        
        # try using different optimizers and different optimizer configs
#        model.compile(loss='binary_crossentropy',
#                      optimizer='adam',
#                      metrics=['accuracy'])
        model.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])
        
        return model

    def lstm_ucf101(self):
        """Build a simple LSTM network. We pass the extracted features from
        CNN pre-trained on imagenet to this model predomenently.
        Starting version from:
        https://github.com/harvitronix/five-video-classification-methods
        """
        # Model.
        model = Sequential()
        model.add(LSTM(2048, return_sequences=True, input_shape=self.input_shape,
                       dropout=0.5))
        model.add(Flatten())
        model.add(Dense(512, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(self.nb_classes, activation='softmax'))
        # Set the metrics. 
        metrics = ['accuracy', 'top_k_categorical_accuracy']
        # Now compile the network.
        optimizer = Adam(lr=1e-6)  # aggressively small learning rate
        self.model.compile(loss='categorical_crossentropy', optimizer=optimizer,
                           metrics=metrics)
        return model
        
        
        