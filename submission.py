#!/usr/bin/env python3
"""Generates submission"""
import pickle
from os.path import join
import numpy as np
from utils import load_data
import pandas as pd
import argparse
from train_lgbm import _mean

# PREDS_DIR = "predictions/"
MODELS_DIR = "models/LGBMs"
DEFAULT_PREPROCESSED_ROOT = "data/features/test"

FEATURE_TRAIN = "data/features/train_overlap"
# DEFAULT_SUBMISSION_FILE = "submission/submission.csv"
DEFAULT_SUBMISSION_FILE = "submission/submission_train_overlap_2.csv"

N_FOLDS = 10
N_CLASSES = 4
CLASSES = ["Normal", "Benign", "InSitu", "Invasive"]

def makeFeature(files, scores):
    X = []
    y = []
    my_dict = {}
    
    for indx, f in enumerate(files):
        f = f.split("_")[0]
        if f in my_dict:
            # print(my_dict[f])
            my_dict[f].append(scores[indx])
        else:
            my_dict[f] = [scores[indx]]

    for k,v in my_dict.items():
        # num_n = my_dict[k].count(0)/ len(my_dict[k])
        # num_b = my_dict[k].count(1)/ len(my_dict[k])
        # num_is = my_dict[k].count(2)/ len(my_dict[k])
        # num_iv = my_dict[k].count(3)/ len(my_dict[k])

        X.append(v)
        if(k[0] == 'n'):
            y.append(0)
        elif(k[0] == 'b'):
            y.append(1)
        elif(k[0]+k[1] == 'is'):
            y.append(2)
        elif(k[0]+k[1] == 'iv'):
            y.append(3)
        else:
            y.append(k)
    
    X = np.stack(X)
    X = np.reshape(X, (len(X),-1))
    y = np.stack(y)

    return X,y

if __name__ == "__main__":

    PREPROCESSED_ROOT = DEFAULT_PREPROCESSED_ROOT
    SUBMISSION_FILE = DEFAULT_SUBMISSION_FILE

    scores = []
    files = None
    len_x = None
    for fold in range(N_FOLDS):
        x, fl = load_data(PREPROCESSED_ROOT)
        if files is None:
            files = fl
            len_x = len(x)
        else:
            np.testing.assert_array_equal(fl, files)
            assert len(x) == len_x
        model_file = "lgbm-f{}.pkl".format(fold)
        with open(join(MODELS_DIR, model_file), "rb") as f:
            model = pickle.load(f)

        sc = model.predict(x)
        sc = sc.reshape(-1, 1, N_CLASSES)
        scores.append(sc)

    scores = np.stack(scores)   # N_FOLDS*N_MODELS*N_SEEDS x N x AUGMENTATIONS_PER_IMAGE x N_CLASSES
    scores = scores.mean(axis=(0, 2))
    # print(scores)
    y_pred = np.argmax(scores, axis=1)
    # print(y_pred)
    # labels = [CLASSES[i] for i in y_pred]

    # df = pd.DataFrame(list(zip(map(lambda s: s.replace(".npy", ".tif"), files), labels, scores)), columns=["image", "label", "probability"])
    # df = df.sort_values("image")
    # df.to_csv(SUBMISSION_FILE, header=False, index=False)
    
    test,test_files = makeFeature(files, scores)
    print(test.shape)
    # print(test)
    # print(test_files)
    # for indx, f in enumerate(test_files):
    #     print(f + ":" + str(test[indx]))

    # for fold in range(N_FOLDS):
    #     preds_file = "lgbm_preds-f{}.pkl".format(fold)
    #     with open(join(PREDS_DIR, preds_file), "rb") as f:
    #         preds = pickle.load(f)

    #     mean_scores = _mean(preds["scores"], mode="arithmetic")
    #     y_pred_lightgbm = np.argmax(mean_scores, axis=1)

    # # X_train, y_train = makeFeature(preds["files"], y_pred_lightgbm)

    scores = []
    files = None
    len_x = None
    for fold in range(N_FOLDS):
        x, fl = load_data(FEATURE_TRAIN)
        if files is None:
            files = fl
            len_x = len(x)
        else:
            np.testing.assert_array_equal(fl, files)
            assert len(x) == len_x
        model_file = "lgbm-f{}.pkl".format(fold)
        with open(join(MODELS_DIR, model_file), "rb") as f:
            model = pickle.load(f)

        sc = model.predict(x)
        sc = sc.reshape(-1, 1, N_CLASSES)
        scores.append(sc)

    scores = np.stack(scores)   # N_FOLDS*N_MODELS*N_SEEDS x N x AUGMENTATIONS_PER_IMAGE x N_CLASSES
    scores = scores.mean(axis=(0, 2))
    y_pred = np.argmax(scores, axis=1)
    
    X_train, y_train = makeFeature(files, scores)
    print(X_train.shape)

    from sklearn import svm

    clf = svm.SVC()
    clf.fit(X_train, y_train)

    result = clf.predict(test)
    labels = [CLASSES[i] for i in result]
    
    df = pd.DataFrame(list(zip(test_files, labels)), columns=["image", "label"])
    df = df.sort_values("image")
    df.to_csv(SUBMISSION_FILE, header=False, index=False)
    