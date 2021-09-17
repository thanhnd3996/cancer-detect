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
from load_datasets import prepare_synthetic_data
from synthetic_models import func_getDecision_Scores_synthetic

import tensorflow as tf
import sys

from itertools import zip_longest
import csv
import shutil
from os.path import basename, join, exists
from keras import backend as K

FEATURE_PATH = 'data/features/train'
DEST = 'data/features_final/train'

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

def nnScore(X, w, V, g):
    X = tf.cast(X, tf.float32)
    w = tf.cast(w, tf.float32)
    V = tf.cast(V, tf.float32)
    return tf.matmul(g((tf.matmul(X, w))), V)

# w_1 = tf.get_variable('Variable', shape=(2048,32))

def relu(x):
    y = x
#     y[y < 0] = 0
    return y

g   = lambda x : relu(x)

def oc_nn_eval():
    
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

    tf.reset_default_graph()

    sess = tf.Session()
    
    new_saver = tf.train.import_meta_graph('save/my_test_model.meta')
    new_saver.restore(sess, tf.train.latest_checkpoint('save/'))
    print(tf.global_variables())
    # print(sess.run('Variable:0'))

    w_1 = sess.run('Variable:0')
    w_2 = sess.run('Variable_1:0')

    rvalue = nnScore(train, w_1, w_2, g)

    with sess.as_default():
        rvalue = rvalue.eval()
        rvalue = np.percentile(rvalue,q=100*0.04)

    train = nnScore(train, w_1, w_2, g)
    test = nnScore(test, w_1, w_2, g)

    with sess.as_default():
        arrayTrain = train.eval()
        arrayTest = test.eval()
        
    sess.close()
    
    rstar = rvalue
    print(rstar)
    
    

    pos_decisionScore = arrayTrain-rstar
    neg_decisionScore = arrayTest-rstar

    n_features = []
    b_features = []
    is_features = []
    iv_features = []

    for indx, normal in enumerate(pos_decisionScore):
        if(normal[0] >= 0):
            n_features.append(train_list[indx])
        
    for indx, abnormal in enumerate(neg_decisionScore):
        if(abnormal[0] < 0):
            if test_list[indx][0] == 'b':
                b_features.append(test_list[indx])
            elif test_list[indx][0] + test_list[indx][1] == 'is':
                is_features.append(test_list[indx])
            else:
                iv_features.append(test_list[indx])

    K.clear_session()

    return [n_features, b_features, is_features, iv_features]

if __name__ == "__main__":

    # print(len(pos_decisionScore))

    from feature_extractor import extract_feature
    from patch_extractor import extract_patch

    first_eval = oc_nn_eval()

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
            print('---copy file: ' + filename)
            dest = join(DEST, filename)
            shutil.copy2(src, dest)
        else:
            print('---copy file: ' + f)
            dest = join(DEST,f)
            shutil.copy2(src, dest)

    for f in first_eval[0]:
        shutil.copy2(join(FEATURE_PATH, f), DEST)
        n_num = n_num + 1

    for f in first_eval[1]:
        copy_new_feature(f, b_num)
        b_num = b_num + 1
    
    for f in first_eval[2]:
        copy_new_feature(f, is_num)
        is_num = is_num + 1

    for f in first_eval[3]:
        copy_new_feature(f, iv_num)
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

        temp_eval = oc_nn_eval()
        
        for f in temp_eval[1]:
            if b_num > n_num:
                break
            copy_new_feature(f, b_num)
            b_num = b_num + 1

        for f in temp_eval[2]:
            if is_num > n_num:
                break
            copy_new_feature(f, is_num)
            is_num = is_num + 1

        for f in temp_eval[3]:
            if iv_num > n_num:
                break
            copy_new_feature(f, iv_num)
            iv_num = iv_num + 1
            
        print(n_num, b_num, is_num, iv_num)

        # dest = 'data/features_final/train/Normal'
        # source = 'data/features/train'
        # for f in n_features:
        #     print(f)
        #     shutil.move(join(source,f) , dest)

        # dest = 'data/features_final/train/Benign'
        # source = 'data/features/train'
        # for f in b_features:
        #     print(f)
        #     shutil.move(join(source,f) , dest)


        # write_decisionScores2Csv("","OC-NN_Linear-load-model.csv",pos_decisionScore,neg_decisionScore)
        # print(pos_decisionScore)

        # import the necessary packages
        
                