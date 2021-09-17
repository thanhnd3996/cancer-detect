#!/usr/bin/env python3
import numpy as np
import pandas as pd
from sklearn import utils
import matplotlib
from keras.preprocessing.image import img_to_array
import cv2
import os
from keras.models import load_model
from keras.models import Model
import keras
from keras.layers import Dense, GlobalAveragePooling2D,Activation

nu = 0.001
# dataPath = './data/'
dataPath = "data/"
from keras import backend as K

import matplotlib.pyplot as plt
from matplotlib import cm
from sklearn.datasets import make_blobs
from load_datasets import prepare_synthetic_data
from synthetic_models import func_getDecision_Scores_synthetic

import tensorflow as tf
import sys

from itertools import zip_longest
import csv
from sklearn import svm

FEATURE_PATH = 'data/features'

def write_decisionScores2Csv(path, filename, positiveScores, negativeScores):

    newfilePath = path+filename
    print ("Writing file to ", path+filename)
    poslist = positiveScores.tolist()
    neglist = negativeScores.tolist()

    # rows = zip(poslist, neglist)
    d = [poslist, neglist]
    export_data = zip_longest(*d, fillvalue='')
    with open(newfilePath, 'w') as myfile:
        wr = csv.writer(myfile)
        wr.writerow(("Normal", "Anomaly"))
        wr.writerows(export_data)
    myfile.close()

    return

def recursive_glob(root_dir, file_template="*.npy"):
    """Traverse directory recursively. Starting with Python version 3.5, the glob module supports the "**" directive"""

    if sys.version_info[0] * 10 + sys.version_info[1] < 35:
        import fnmatch
        import os
        matches = []
        for root, dirnames, filenames in os.walk(root_dir):
            for filename in fnmatch.filter(filenames, file_template):
                matches.append(os.path.join(root, filename))
        return matches
    else:
        import glob
        return glob.glob(root_dir + "/**/" + file_template, recursive=True)

if __name__ == "__main__":
    # [syn_train,syn_train_label,syn_test,syn_test_label] = prepare_synthetic_data()
    # print ("[INFO] SYNTHETIC DATA : (Normal, Anomalies) dimensions: ",syn_train.shape,syn_test.shape)
    # print(syn_train.shape, syn_test.shape)
    # print(syn_train_label.shape, syn_test_label.shape)
    # print(syn_test, syn_test_label)
    
    train = []
    train_label = []
    test = []
    test_label = []

    train_list = []
    test_list = []

    input_files = recursive_glob(FEATURE_PATH)
    # print(input_files)

    for f in input_files:
        x = np.load(f)
        # print(f)
        class_name = f.split("/")[3]
        if(class_name[0] == 'n'):
            train.append(x[0])
            train_label.append(0.)
            train_list.append(class_name)
        else:
            test.append(x[0])
            test_label.append(1.)
            test_list.append(class_name)

    train = np.stack(train)
    test = np.stack(test)
    train_label = np.stack(train_label)
    test_label = np.stack(test_label)

    ocSVM = svm.OneClassSVM(nu = 0.001, kernel = 'rbf')
    ocSVM.fit(train)

    # pos_decisionScore = ocSVM.decision_function(train)
    # neg_decisionScore = ocSVM.decision_function(test)

    pos_decisionScore = ocSVM.predict(train)
    neg_decisionScore = ocSVM.predict(test)

    # print(pos_decisionScore)
    write_decisionScores2Csv("","123-predict.csv", pos_decisionScore, neg_decisionScore)
    # print(train_list)
    # print(test_list)
    # test = test[:20]
    # test_label = test_label[:20]
    # train_label = np.concatenate((y, y_a), axis=0)
    # print(train.shape, test.shape)
    # print(train_label.shape, test_label.shape)
    # print(test, test_label)
    # df_syn_scores = func_getDecision_Scores_synthetic(syn_train,syn_train,syn_test,syn_train_label,autoencoder="no")
    # print (syn_test)
    
    # df_syn_scores = func_getDecision_Scores_synthetic(train,train,test,train_label,autoencoder="no")
         