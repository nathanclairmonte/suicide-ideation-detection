"""
This file performs the following steps:
    1. Loads data from CSV
    2. Extracts raw text data
    3. Processes text data (this involves cleaning, stemming, lemmatization,
       stop-word removal etc.)
    4. Splits text data into training and testing sets
    5. Creates Tf-Idf features from the text data (final ngram range used was (1, 2))
    6. Saves the features to file for easy loading when running experiments.
"""

import os
import pandas as pd
import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import scipy
import helperFunctions as hf

# CONSTANTS

# data folder location
curr = os.path.dirname(__file__)
DATA_FOLDER = os.path.join(curr, "../data/")

# max_df for tf-idf features
MAX_DF = 0.1

# ngram range for tf-idf features
NGRAM_RANGE = (1, 2)

# whether or not to print while running
VERBOSE = True

if (__name__=="__main__"):
    # load data and create labels column
    data = pd.read_csv(DATA_FOLDER + "Suicide_Detection_Data.csv")
    data['labels'] = np.where(data['class']=='suicide', 1, 0)

    # extract raw text and labels
    X_raw = list(data['text'])
    y = data['labels'].values

    # preprocess text
    t_start = time.time()
    X = hf.preprocess_doc(X_raw,
                          stemming=True,
                          lemmatization=False,
                          stopwords=False)
    t_end = time.time()

    if VERBOSE:
        print(f"Time elapsed: {hf.stringTime(t_start, t_end)}\n")
        print(f"{len(X):,} samples processed")

    # split into train/test sets
    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        train_size=0.8,
                                                        test_size=0.2)
    if VERBOSE:
        print("X_train: ", len(X_train))
        print("y_train: ", len(y_train))
        print("X_test:  ", len(X_test))
        print("y_test:  ", len(y_test))

    # tf-idf features
    vect = TfidfVectorizer(max_df=MAX_DF,
                           strip_accents='unicode',
                           ngram_range=NGRAM_RANGE)
    t_start = time.time()
    X_train = vect.fit_transform(X_train)
    X_test = vect.transform(X_test)
    t_end = time.time()

    if VERBOSE:
        print(f"Time elapsed: {hf.stringTime(t_start, t_end)}\n")
        print(f"X_train: {X_train.shape}")
        print(f"X_test:  {X_test.shape}")

    # save to file
    X_train_path = DATA_FOLDER + "X_train.npz"
    X_test_path = DATA_FOLDER + "X_test.npz"
    y_train_path = DATA_FOLDER + "y_train.npy"
    y_test_path = DATA_FOLDER + "y_test.npy"
    scipy.sparse.save_npz(X_train_path, X_train)
    scipy.sparse.save_npz(X_test_path, X_test)
    np.save(y_train_path, y_train)
    np.save(y_test_path, y_test)