"""
This file performs the following steps:
    1. Loads train and test data from npz/npy files
    2. Trains a baseline model and calculates the false negative rate (FNR) of predictions
    3. Trains a model with decreased prediction threshold and calculates FNR of predictions
    4. Trains a model with altered class weights and calculates FNR of predictions
    5. Compares FNR values and logs FNR reduction from each of the two FNR reduction methods.
"""

import os
from scipy import sparse
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
import helperFunctions as hf

# CONSTANTS

# data folder location
curr = os.path.dirname(__file__)
DATA_FOLDER = os.path.join(curr, "../data/")

# reduced thresh value for the first FNR reduction method
REDUCED_THRESHOLD = -0.1

# whether or not to print while running
VERBOSE = True

if (__name__=="__main__"):
    # load data
    X_train_filename = DATA_FOLDER + "X_train.npz"
    X_test_filename = DATA_FOLDER + "X_test.npz"
    y_train_filename = DATA_FOLDER + "y_train.npy"
    y_test_filename = DATA_FOLDER + "y_test.npy"
    X_train = sparse.load_npz(X_train_filename)
    X_test = sparse.load_npz(X_test_filename)
    y_train = np.load(y_train_filename)
    y_test = np.load(y_test_filename)

    if VERBOSE:
        print(f"X_train: ({X_train.shape[0]:,}, {X_train.shape[1]:,})")
        print(f"y_train: ({len(y_train):,},)")
        print(f"X_test:  ({X_test.shape[0]:,}, {X_test.shape[1]:,})")
        print(f"y_test:  ({len(y_test):,},)\n\n")

    # -------------- baseline model --------------
    # logging
    if VERBOSE:
        print("-------------- Baseline Model --------------\n")

    # create model
    clf = LinearSVC(random_state=1)

    # train model
    clf.fit(X_train, y_train)

    # predict
    y_pred = clf.predict(X_test)

    # evaluate
    acc_baseline = accuracy_score(y_test, y_pred)
    f1_baseline = f1_score(y_test, y_pred)
    pre_baseline = precision_score(y_test, y_pred)
    recall_baseline = recall_score(y_test, y_pred)

    # calculate FNR
    cm = confusion_matrix(y_test, y_pred)
    TP_baseline = cm[1][1]
    FN_baseline = cm[1][0]
    FNR_baseline = FN_baseline/(TP_baseline + FN_baseline)

    # logging
    if VERBOSE:
        print(f"Accuracy : {acc_baseline*100:.3f}%")
        print(f"F1 Score : {f1_baseline*100:.3f}%")
        print(f"Precision: {pre_baseline*100:.3f}%")
        print(f"Recall   : {recall_baseline*100:.3f}%\n")
        print(f"TP: {TP_baseline:,}")
        print(f"FN: {FN_baseline:,}\n")
        print(f"FNR: {FNR_baseline*100:.3f}%\n\n")

    
    # -------------- decreased prediction threshold --------------
    # logging
    if VERBOSE:
        print("-------------- Decreased Prediction Threshold --------------\n")

    # create model
    clf = LinearSVC(random_state=1)

    # train model
    clf.fit(X_train, y_train)

    # get confidence scores
    y_conf = clf.decision_function(X_test)

    # build y_pred using threshold
    y_pred = np.where(y_conf > REDUCED_THRESHOLD, 1, 0)

    # evaluate
    acc_dec_thresh = accuracy_score(y_test, y_pred)
    f1_dec_thresh = f1_score(y_test, y_pred)
    pre_dec_thresh = precision_score(y_test, y_pred)
    recall_dec_thresh = recall_score(y_test, y_pred)

    # calculate FNR
    cm = confusion_matrix(y_test, y_pred)
    TP_dec_thresh = cm[1][1]
    FN_dec_thresh = cm[1][0]
    FNR_dec_thresh = FN_dec_thresh/(TP_dec_thresh + FN_dec_thresh)

    # logging
    if VERBOSE:
        print(f"Accuracy : {acc_dec_thresh*100:.3f}%")
        print(f"F1 Score : {f1_dec_thresh*100:.3f}%")
        print(f"Precision: {pre_dec_thresh*100:.3f}%")
        print(f"Recall   : {recall_dec_thresh*100:.3f}%\n")
        print(f"TP: {TP_dec_thresh:,}")
        print(f"FN: {FN_dec_thresh:,}\n")
        print(f"FNR: {FNR_dec_thresh*100:.3f}%\n\n")

    
    # -------------- altered class weights --------------
    # logging
    if VERBOSE:
        print("-------------- Altered Class Weights --------------\n")

    # create model
    weights = {0:1.0, 1:100.0}
    clf = LinearSVC(random_state=1, class_weight=weights)

    # train model
    clf.fit(X_train, y_train)

    # predict
    y_pred = clf.predict(X_test)

    # evaluate
    acc_alt_cw = accuracy_score(y_test, y_pred)
    f1_alt_cw = f1_score(y_test, y_pred)
    pre_alt_cw = precision_score(y_test, y_pred)
    recall_alt_cw = recall_score(y_test, y_pred)

    # calculate FNR
    cm = confusion_matrix(y_test, y_pred)
    TP_alt_cw = cm[1][1]
    FN_alt_cw = cm[1][0]
    FNR_alt_cw = FN_alt_cw/(TP_alt_cw + FN_alt_cw)

    # logging
    if VERBOSE:
        print(f"Accuracy : {acc_alt_cw*100:.3f}%")
        print(f"F1 Score : {f1_alt_cw*100:.3f}%")
        print(f"Precision: {pre_alt_cw*100:.3f}%")
        print(f"Recall   : {recall_alt_cw*100:.3f}%\n")
        print(f"TP: {TP_alt_cw:,}")
        print(f"FN: {FN_alt_cw:,}\n")
        print(f"FNR: {FNR_alt_cw*100:.3f}%\n\n")


    # -------------- FNR comparison --------------
    print("-------------- FNR Comparison --------------\n")
    print(f"Baseline: {FNR_baseline*100:.3f}%")
    print(f"Decreased Pred Threshold: {FNR_dec_thresh*100:.3f}%")
    print(f"Altered Class Weights: {FNR_alt_cw*100:.3f}%\n")

    print("Decreasing the prediction threshold resulted in a:")
    print(f"    - {hf.percentChange(FNR_baseline, FNR_dec_thresh):.1f}% decrease in FNR")
    print(f"    - {hf.percentChange(acc_baseline, acc_dec_thresh):.1f}% decrease in overall accuracy\n")
    
    print("Altering the class weights resulted in a:")
    print(f"    - {hf.percentChange(FNR_baseline, FNR_alt_cw):.1f}% decrease in FNR")
    print(f"    - {hf.percentChange(acc_baseline, acc_alt_cw):.1f}% decrease in overall accuracy")
