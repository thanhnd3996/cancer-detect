from sklearn import utils
import matplotlib
from keras.preprocessing.image import img_to_array
import cv2
import os
from keras.models import load_model
from keras.models import Model
import keras
from keras.layers import Dense, GlobalAveragePooling2D,Activation

from keras import backend as K
import tensorflow as tf
import sys

from itertools import zip_longest
import csv
import shutil
from os.path import basename, join, exists
from keras import backend as K
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from feature_extractor import recursive_glob
import numpy as np

FEATURE_PATH = 'data/features/train_overlap'

if __name__ == "__main__":
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
        # class_name = f.split("/")[3]
        class_name = basename(f)
        if(class_name[0] == 'n'):
            train.append(x[0])
            train_label.append(1)
            train_list.append(class_name)
        else:
            test.append(x[0])
            test_label.append(-1)
            test_list.append(class_name)

    train = np.stack(train)
    test = np.stack(test)
    # print(train.shape)
    # print(test.shape)
    train_label = np.stack(train_label)
    test_label = np.stack(test_label)
    
    # print(train.shape)
    # ocSVM = svm.OneClassSVM(nu = 0.05, kernel = 'rbf')
    # ocSVM.fit(train)

    nu = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09]
    gammas = [0.001, 0.005 ,0.01, 0.05, 0.1, 1/2048]
    param_grid = {'nu': nu, 'gamma' : gammas}
    grid_search = GridSearchCV(svm.OneClassSVM(kernel='rbf'), param_grid, scoring='accuracy', cv=10)
    grid_search.fit(train,train_label)
    # grid_search.best_params_
    # print (grid_search.best_params_)
    print (grid_search.cv_results_)
    print (grid_search.best_estimator_.n_support_)
    print (len(grid_search.best_estimator_.n_support_))

    # pos_decisionScore = ocSVM.decision_function(train)
    # neg_decisionScore = ocSVM.decision_function(test)

    # pos_decisionScore = ocSVM.predict(train)
    
    # neg_decisionScore = ocSVM.predict(test)
    # print(np.count_nonzero(pos_decisionScore == 1))
    # print(np.count_nonzero(neg_decisionScore == -1))

    # n_features = []
    # b_features = []
    # is_features = []
    # iv_features = []

    # for indx, normal in enumerate(pos_decisionScore):
    #     if(normal >= 0):
    #         n_features.append(train_list[indx])
        
    # for indx, abnormal in enumerate(neg_decisionScore):
    #     if(abnormal < 0):
    #         if test_list[indx][0] == 'b':
    #             b_features.append(test_list[indx])
    #         elif test_list[indx][0] + test_list[indx][1] == 'is':
    #             is_features.append(test_list[indx])
    #         else:
    #             iv_features.append(test_list[indx])

    # return [n_features, b_features, is_features, iv_features]