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

dataPath = "data/"
from keras import backend as K

from sklearn.datasets import make_blobs

import tensorflow as tf
import sys

from itertools import zip_longest
import csv
import shutil
from os.path import basename, join, exists
from keras import backend as K
from sklearn import svm
from os import makedirs
# FEATURE_PATH = 'data/features/train_overlap'
FEATURE_PATH = 'data/features/train'

DEST = 'data/features_one_class/train'

RETRAIN_DIR = 'data/retrain'
PREPROCESS_DIR = 'data/preprocessed/train'

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

def oc_svm_eval():
    
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
            train_label.append(1.)
            train_list.append(class_name)
        else:
            test.append(x[0])
            test_label.append(-1.)
            test_list.append(class_name)

    train = np.stack(train)
    test = np.stack(test)
    # print(train.shape)
    # print(test.shape)
    train_label = np.stack(train_label)
    test_label = np.stack(test_label)
    
    # print(train.shape)
    # ocSVM = svm.OneClassSVM(nu = 0.05, kernel = 'rbf')
    ocSVM = svm.OneClassSVM(nu = 0.05, kernel = 'rbf')
    ocSVM.fit(train)

    # pos_decisionScore = ocSVM.decision_function(train)
    # neg_decisionScore = ocSVM.decision_function(test)

    pos_decisionScore = ocSVM.predict(train)
    neg_decisionScore = ocSVM.predict(test)
    # print(np.count_nonzero(pos_decisionScore == 1))
    # print(np.count_nonzero(neg_decisionScore == -1))

    n_features = []
    b_features = []
    is_features = []
    iv_features = []

    for indx, normal in enumerate(pos_decisionScore):
        if(normal >= 0):
            n_features.append(train_list[indx])
        
    for indx, abnormal in enumerate(neg_decisionScore):
        if(abnormal < 0):
            if test_list[indx][0] == 'b':
                b_features.append(test_list[indx])
            elif test_list[indx][0] + test_list[indx][1] == 'is':
                is_features.append(test_list[indx])
            else:
                iv_features.append(test_list[indx])

    return [n_features, b_features, is_features, iv_features]

if __name__ == "__main__":

    # print(len(pos_decisionScore))

    from feature_extractor import extract_feature
    from patch_extractor import extract_patch

    first_eval = oc_svm_eval()

    n_num = 0
    b_num = 0
    is_num = 0
    iv_num = 0
    
    #copying new feature avoid same name
    def copy_new_feature(f, num):
        s = list(f)
        filename = s[:len(s)-4]
        filename.append("_"+str(num))
        filename = "".join(filename)
        filename = filename + '.npy'
        
        src = join(FEATURE_PATH, f)
        if(exists(join(DEST,f))):
            # print('---copy file: ' + filename)
            dest = join(DEST, filename)
            shutil.copy2(src, dest)
        else:
            # print('---copy file: ' + f)
            dest = join(DEST,f)
            shutil.copy2(src, dest)

    #copying patches 
    def copy_new_patch(f, num):
        s = list(f)
        filename = s[:len(s)-4]
        # filename.append("_"+str(num))
        # filename = "".join(filename)
        # filename = filename + '.npy'
        
        if(filename[0] == 'n'):
            filename[0] = 't'
            class_name = 'Normal'
        elif (filename[0] == 'b'):
            filename[0] = 't'
            class_name = 'Benign'
        elif (filename[0] + filename[1] == 'is'):
            filename[1] = 't'
            filename.pop(0)
            class_name = 'In Situ'
        elif (filename[0] + filename[1] == 'iv'):
            filename[1] = 't'
            filename.pop(0)
            class_name = 'Invasive'

        filename = "".join(filename)
        filename_with_index = filename +"_"+str(num)

        filename = filename + '.npy'
        filename_with_index = filename_with_index + '.npy'

        src = join(PREPROCESS_DIR, class_name, filename)
        # print(src)
        
        if not exists(join(RETRAIN_DIR,class_name)):
            makedirs(join(RETRAIN_DIR,class_name))

        if(exists(join(RETRAIN_DIR,class_name,filename))):
            # print('---copy file: ' + filename)
            dest = join(RETRAIN_DIR,class_name,filename_with_index)
            shutil.copy2(src, dest)
        else:
            # print('---copy file: ' + f)
            dest = join(RETRAIN_DIR,class_name,filename)
            # print(dest)
            shutil.copy2(src, dest)

    for f in first_eval[0]:
        copy_new_feature(f, n_num)
        # copy_new_patch(f, n_num)
        n_num = n_num + 1

    for f in first_eval[1]:
        copy_new_feature(f, b_num)
        # copy_new_patch(f, b_num)
        b_num = b_num + 1
    
    for f in first_eval[2]:
        copy_new_feature(f, is_num)
        # copy_new_patch(f, is_num)
        is_num = is_num + 1

    for f in first_eval[3]:
        copy_new_feature(f, iv_num)
        # copy_new_patch(f, iv_num)
        iv_num = iv_num + 1
    
    print(n_num, b_num, is_num, iv_num)

    while( n_num > b_num or n_num > is_num or n_num > iv_num ):
        classes = ['Benign', 'Invasive', 'In Situ']
        if(b_num >= n_num):
            classes.remove('Benign')
        if(iv_num >= n_num):
            classes.remove('Invasive')
        if(is_num >= n_num):
            classes.remove('In Situ')

        extract_patch(classes)
        extract_feature(classes)

        temp_eval = oc_svm_eval()
        
        for f in temp_eval[1]:
            if b_num >= n_num:
                break
            copy_new_feature(f, b_num)
            # copy_new_patch(f, b_num)
            b_num = b_num + 1

        for f in temp_eval[2]:
            if is_num >= n_num:
                break
            copy_new_feature(f, is_num)
            # copy_new_patch(f, is_num)
            is_num = is_num + 1

        for f in temp_eval[3]:
            if iv_num >= n_num:
                break
            copy_new_feature(f, iv_num)
            # copy_new_patch(f, iv_num)
            iv_num = iv_num + 1
            
        print(n_num, b_num, is_num, iv_num)

        
                