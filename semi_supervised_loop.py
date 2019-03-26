# -*- coding: utf-8 -*-
"""
Created on Sat Jan 27 16:54:58 2018

@author: Li Zhun

This script will perform loop of semi-supervised deep learning.
"""

###Reproducibility##############################################################
import numpy as np
#import tensorflow as tf
#import random as rn
#
#import os
#os.environ['PYTHONHASHSEED'] = '0'
#
#np.random.seed(42)
#rn.seed(12345)
#session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
#
#from keras import backend as K
#
#tf.set_random_seed(1234)
#sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
#K.set_session(sess)
###End##########################################################################

from DataSets import DataSets
from PseudoLabelingModels import PseudoLabelingModels
from DeepLearningModels import DeepLearningModels

import sys
import datetime as dt
import time
from sklearn import metrics

from keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger #,TensorBoard
from keras.utils import to_categorical
from keras.models import load_model
from keras.backend import clear_session
import pickle


def semi_supervised_loop(num_loop, labeling_model_name, main_model_name, num_classes=10, 
                         sample_per_class = None, saved_model=None, balanced=False):
    """
    `labeling_model_name` = one of:svc, gpc, rf(random forest), or deep learning models
    `main_model_name` = one of: mlp_mnist, cnn_cifar10, lstm_imdb, lstm_ucf101
    `data_name` = one of: mnist, cifar10, imdb, ucf101
    `num_classes` = the number of classes to predict
    `sample_per_class` = the sampling number of each class, None means all.
    """

    if saved_model == None:
        # Get the appropriate data.
        (x_tr, y_tr), (x_test, y_test), (x_unl, y_unl) = DataSets(main_model_name, sample_per_class).data

#        # Use for analyzing the factor of unlabeled data size
#        num_unlabeled = 10000
#        unindex = np.random.choice(len(x_unl), num_unlabeled, replace=False)
#        x_unl = x_unl[unindex]
#        y_unl = y_unl[unindex]

        # when labeling_model is not deep learning model
        if labeling_model_name == 'random':
            print("Pseudo Labeling with random number.")
            y_pseudo = np.random.choice(range(num_classes),len(y_unl))
            print("Pseudo Labeling Accuracy:")
            print(metrics.accuracy_score(y_unl, y_pseudo))
    
        elif labeling_model_name == 'truth':
            print("Pseudo Labeling with ground truth.")
            y_pseudo = y_unl
            print("Pseudo Labeling Accuracy:")
            print(metrics.accuracy_score(y_unl, y_pseudo))
    
        elif labeling_model_name in ['svm','gpc','rf']:
            # Get the appropriate model.
            labeling_model = PseudoLabelingModels(num_classes, labeling_model_name).model
            
            #reshape data for non-deep learning model
            x_tr_r = x_tr.reshape((x_tr.shape[0],-1))
            x_test_r = x_test.reshape((x_test.shape[0],-1))
            x_unl_r = x_unl.reshape((x_unl.shape[0],-1))
                    
            #train labeling_model, count time
            start_time = dt.datetime.now()
            print('Start training pseudo-labeling model at {}'.format(str(start_time)))
            labeling_model.fit(x_tr_r, y_tr)
            end_time = dt.datetime.now() 
            print('Stop training at {}'.format(str(end_time)))
            elapsed_time= end_time - start_time
            print('Elapsed {}'.format(str(elapsed_time)))
            
            # Evaluation on test data set
            y_test_pred = labeling_model.predict(x_test_r)
            print("Pseudo Labeling model Accuracy on test data set:")
            print(metrics.accuracy_score(y_test, y_test_pred))

            # PseudoLabeling
            y_pseudo = labeling_model.predict(x_unl_r)
            # show PseudoLabeling accurary
            print("Pseudo Labeling Accuracy:")
            print(metrics.accuracy_score(y_unl, y_pseudo))
    
#        #split tr to val and new tr
#        val_percent = 0.2
#        val_num = int(round(sample_per_class*val_percent, 0))
#        sample_index_val = DataSets.random_sampling_by_class(val_num, y_tr)
#        x_val = x_tr[sample_index_val]
#        y_val = y_tr[sample_index_val]
#        #compute the complementary set of sample_index
#        sample_index_tr = list(set(np.arange(len(y_tr))).difference(set(sample_index_val)))
#        x_tr = x_tr[sample_index_tr]
#        y_tr = y_tr[sample_index_tr]
        x_val = x_test[0:100]
        y_val = y_test[0:100]
           
        model = DeepLearningModels(num_classes, main_model_name).model
        # convert class vectors to binary class matrices
        y_tr = to_categorical(y_tr, num_classes)
        y_val = to_categorical(y_val, num_classes)
        y_test = to_categorical(y_test, num_classes)

        #use deep learning model itselt as pseudo-labeling model
        if labeling_model_name in ['mlp_mnist','cnn_mnist','cnn_cifar10','cnn_cifar100',
                                   'lstm_imdb','lstm_ucf101']:
            # PseudoLabeling
            model,y_unl_proba = train_func(x_tr, y_tr, x_test, y_test, x_unl, to_categorical(y_unl, num_classes), 
                                  model, main_model_name, num_classes, patience=10, x_val=x_val, y_val=y_val)
            y_pseudo = np.argmax(y_unl_proba, axis=1)
            # show PseudoLabeling accurary
            print("Pseudo Labeling Accuracy:")
            print(metrics.accuracy_score(y_unl, y_pseudo))
    else:
        print('Loading saved data.')
        with open(saved_model + '.pickle', 'rb') as data:
            (y_pseudo,(x_tr, y_tr), (x_val, y_val), (x_test, y_test), (x_unl, y_unl)) = pickle.load(data)
        print('Loading saved model.')
        model = load_model(saved_model)
        # show PseudoLabeling accurary
        print("Pseudo Labeling Accuracy:")
        print(metrics.accuracy_score(y_unl, y_pseudo))
#        sys.exit()#=============================================================================
        
    for i in range(num_loop):
        if num_loop>1:
            print('{}-----------------------------------------------------------'.format(str(i)))

        if balanced:
            num_sam = int(len(y_pseudo)/num_classes)
            sample_index = DataSets.random_sampling_by_class(num_sam, y_pseudo)
    
            y_pseudo = to_categorical(y_pseudo, num_classes)
    
            print("Start to fit {} with pseudo label data.".format(main_model_name))
            model,_ = train_func(x_unl[sample_index], y_pseudo[sample_index], x_test, y_test, x_test[0:1], y_test[0:1], 
                           model, main_model_name, num_classes, patience=2, x_val=x_val, y_val=y_val)
        else:
            y_pseudo = to_categorical(y_pseudo, num_classes)
    
            print("Start to fit {} with pseudo label data.".format(main_model_name))
            model,_ = train_func(x_unl, y_pseudo, x_test, y_test, x_test[0:1], y_test[0:1], 
                           model, main_model_name, num_classes, patience=2, x_val=x_val, y_val=y_val)
        
        print("Start to fit {} with labeled data.".format(main_model_name))
        model,y_unl_proba = train_func(x_tr, y_tr, x_test, y_test, x_unl, to_categorical(y_unl, num_classes), 
                       model, main_model_name, num_classes, patience=2, x_val=x_val, y_val=y_val)

        y_pseudo = np.argmax(y_unl_proba, axis=1)
        # show PseudoLabeling accurary
        print("Pseudo Labeling Accuracy:")
        print(metrics.accuracy_score(y_unl, y_pseudo))

    filepath='./data/checkpoints/'+ main_model_name + '-' +  labeling_model_name + '-' +  \
    str(sample_per_class) + '.h5'
#    str(sample_per_class) + '-' +  str(num_loop) + '-' + str(time.time()) + '.h5'
    model.save(filepath)
    #save data
    with open(filepath + '.pickle', 'wb') as output:
        pickle.dump((y_pseudo,(x_tr, y_tr), (x_val, y_val), (x_test, y_test), (x_unl, y_unl)), output)

def train_func(x, y, x_test, y_test, x_unl, y_unl, model, model_name, 
               num_classes, patience=10, x_val=None, y_val=None):
    """
    `x, y`: training data
    `x_test, y_test`: test data, to evaluate model
    `x_unl, y_unl`: unlabeled data, to generate prediction, y_unl is useless
    `model`: model object
    `model_name`: the name of model
    `num_classes`: the number of classes to be predicted
    `x_val, y_val`： validation data. If val data is None, train model with fix epoch. 
    """
    if model_name == 'mlp_mnist':
        batch_size = 128
        epochs = 20
    elif model_name == 'cnn_mnist':
        batch_size = 128
        epochs = 20
    elif model_name == 'cnn_cifar10':
        batch_size = 128
        epochs = 100
    elif model_name == 'cnn_cifar100':
        batch_size = 128
        epochs = 100
    elif model_name == 'lstm_imdb':
        batch_size = 32
        epochs = 15
    elif model_name == 'lstm_ucf101':
        batch_size = 128
        epochs = 20
    else:
        print("Unknown model.")
        sys.exit()

    # Helper: Save the model.
    checkpointer = ModelCheckpoint(
        filepath='./data/checkpoints/' + model_name + '.hdf5',
        verbose=0,
        save_best_only=True)

#    # Helper: TensorBoard
#    tb = TensorBoard(log_dir='./data/logs')

    # Helper: Stop when we stop learning.
    early_stopper = EarlyStopping(patience=patience)

    # Helper: Save results.
    timestamp = time.time()
    csv_logger = CSVLogger('./data/logs/' + model_name + '-' + 'training-' + \
        str(timestamp) + '.log')

    if x_val is None:
        model.fit(
            x,
            y,
            batch_size=batch_size,
    #        validation_split=0.2,
#            validation_data=(x_val, y_val),
            verbose=0,
    #        callbacks=[checkpointer, tb, early_stopper, csv_logger],
    #        callbacks=[checkpointer, early_stopper, csv_logger],
            epochs=epochs)
    else:
        model.fit(
            x,
            y,
            batch_size=batch_size,
    #        validation_split=0.2,
            validation_data=(x_val, y_val),
            verbose=1,
    #        callbacks=[checkpointer, tb, early_stopper, csv_logger],
            callbacks=[checkpointer, early_stopper, csv_logger],
            epochs=epochs)
        
        # evaluate model
    #    print('Start to evaluate')    
        clear_session()
        model = load_model('./data/checkpoints/' + model_name + '.hdf5')

    results = model.evaluate(
        x_test,
        y_test,
        batch_size=batch_size,
        verbose=0)
    print('Evaluation results of saved model on test data is:')
    print(results)
    print(model.metrics_names)
    
    return model,model.predict(
        x_unl,
        batch_size=batch_size,
        verbose=0)

    
def main():
    #test reproducibility
#    print(np.random.choice(range(5),10))
#    d=DataSets('mnist', 5)
#    print(d.random_sampling_by_class(2,np.array([1,1,1,1,2,2,2,2,3,3,3,3])))
    
    print('Start running at {}'.format(str(dt.datetime.now())))
    
    num_loop = 10
    repeat_times = 30
    labeling_model_name = 'cnn_mnist'# random,truth,svm,gpc,rf,deep
    main_model_name = 'cnn_mnist'# mlp_mnist, cnn_mnist, cnn_cifar10, lstm_imdb, lstm_ucf101
    num_classes = 10
    sample_per_class = 40 #5,10,50,100,300
    saved_model = None #'./data/checkpoints/mlp_mnist-svm-5-5-1517073670.666765.h5'
#    saved_model = './data/checkpoints/'+ main_model_name + '-' +  labeling_model_name + '-' +  \
#    str(sample_per_class) + '.h5'
    balanced = False
            
    savedStdout = sys.stdout  #save std output
    out_file_name = './result/'+ main_model_name + '-' +  labeling_model_name + '-' +  \
    str(sample_per_class) + '-' +  str(num_loop) + '-' +  str(repeat_times) + '-' + str(time.time()) + '.txt'
    with open(out_file_name, 'w+', buffering = 1) as file:
        for i in range(repeat_times):
            print('Start task {} at {}'.format(str(i),str(dt.datetime.now())))
            sys.stdout = file  #change stdout to file
            start_time = dt.datetime.now()
            print('Start task {} at {}'.format(str(i),str(start_time)))
            semi_supervised_loop(num_loop, labeling_model_name, main_model_name, num_classes, 
                                 sample_per_class, saved_model, balanced)
            clear_session()
            end_time = dt.datetime.now() 
            print('Stop task {} at {}'.format(str(i),str(end_time)))
            elapsed_time= end_time - start_time
            print("Total running time is {} seconds.".format(str(elapsed_time)))
            print('--End-----------------------------------------------------------')
            sys.stdout = savedStdout  #recover std output
    
    print('Stop running at {}'.format(str(dt.datetime.now())))


if __name__ == '__main__':
    main()
