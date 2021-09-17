#!/usr/bin/env python3
"""Trains LightGBM models on various features, data splits. Dumps models and predictions"""

import pickle
import numpy as np
import lightgbm as lgb
from sklearn.metrics import accuracy_score
from utils import load_data
from os.path import join, exists
from os import makedirs
import argparse

AUGMENTATIONS_PER_IMAGE = 50
NUM_CLASSES = 4
RANDOM_STATE = 1
N_SEEDS = 5
VERBOSE_EVAL = False
with open("data/folds-10.pkl", "rb") as f:
    FOLDS = pickle.load(f)

LGBM_MODELS_ROOT = "models/LGBMs"
CROSSVAL_PREDICTIONS_ROOT = "predictions"
DEFAULT_PREPROCESSED_ROOT = "data/features_one_class/train"

# DEFAULT_PREPROCESSED_ROOT = "data/features/train_overlap"


def _mean(x, mode="arithmetic"):
    """
    Calculates mean probabilities across augmented data

    # Arguments
        x: Numpy 3D array of probability scores, (N, AUGMENTATIONS_PER_IMAGE, NUM_CLASSES)
        mode: type of averaging, can be "arithmetic" or "geometric"
    # Returns
        Mean probabilities 2D array (N, NUM_CLASSES)
    """
    assert mode in ["arithmetic", "geometric"]
    if mode == "arithmetic":
        x_mean = x.mean(axis=1)
    else:
        x_mean = np.exp(np.log(x + 1e-7).mean(axis=1))
        x_mean = x_mean / x_mean.sum(axis=1, keepdims=True)
    return x_mean


if __name__ == "__main__":

    PREPROCESSED_ROOT = DEFAULT_PREPROCESSED_ROOT

    learning_rate = 0.3
    num_round = 70
    param = {
        "objective": "multiclass",
        "num_class": NUM_CLASSES,
        "metric": ["multi_logloss", "multi_error"],
        "verbose": 1,
        "learning_rate": learning_rate,
        "num_leaves": 30,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.56,
        "bagging_freq": 70,
        "max_depth": 7,
        "min_data_in_leaf": 100
    }
 
    INPUT_DIR = PREPROCESSED_ROOT
    
    accuracies = []
    for fold in range(len(FOLDS)):
        # feature_fraction_seed = RANDOM_STATE + 1 * 10 + fold
        # bagging_seed = feature_fraction_seed + 1
        # param.update({"feature_fraction_seed": feature_fraction_seed, "bagging_seed": bagging_seed})

        print("Fold {}/{}".format(fold + 1, len(FOLDS)))
        x_train, y_train, x_test, y_test = load_data(INPUT_DIR, FOLDS, fold)
        print(x_train.shape)
        train_data = lgb.Dataset(x_train, label=y_train)
        test_data = lgb.Dataset(x_test, label=y_test)
        gbm = lgb.train(param, train_data, num_round, valid_sets=[test_data], verbose_eval=VERBOSE_EVAL)

        # pickle model
        model_file = "lgbm-f{}.pkl".format(fold)
        model_root = LGBM_MODELS_ROOT
        
        if not exists(model_root):
            makedirs(model_root)
        with open(join(model_root, model_file), "wb") as f:
            pickle.dump(gbm, f)

        scores = gbm.predict(x_test)
        scores = scores.reshape(-1, 1, NUM_CLASSES)

        print('--score2--')
        print(scores)
        print(scores.shape)
        preds = {
            "files": FOLDS[fold]["test"]["x"],
            "y_true": y_test,
            "scores": scores,
        }
        # print(preds)
        preds_file = "lgbm_preds-f{}.pkl".format(fold)
        # preds_file = "lgbm_preds-f{}-overlap.pkl".format(fold)
        preds_root = CROSSVAL_PREDICTIONS_ROOT
        if not exists(preds_root):
            makedirs(preds_root)
        with open(join(preds_root, preds_file), "wb") as f:
            pickle.dump(preds, f)

        mean_scores = _mean(scores, mode="arithmetic")
        print(mean_scores)
        y_pred = np.argmax(mean_scores, axis=1)
        print(y_pred)
        y_true = y_test[::1]
        print(y_test)
        print(y_true)
        acc = accuracy_score(y_true, y_pred)
        print("Accuracy:", acc)
        accuracies.append(acc)

    acc_seed = np.array(accuracies).mean()  # acc of a seed
    print("Accuracies: [{}], mean {:5.3}".format(", ".join(map(lambda s: "{:5.3}".format(s), accuracies)),
                                                            acc_seed))
