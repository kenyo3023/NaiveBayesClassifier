#encoding=utf8
import numpy as np
import pandas as pd
import struct
import os
import random
import matplotlib.pyplot as plt
from random import sample
from sklearn import metrics
from PIL import Image
from argparse import ArgumentParser

from MultinomialNB import MultinomialNB
from GaussianNB import GaussianNB


def load_data(path, data_type):
    y_train = np.load(os.path.join(path, 'y_train.npy')).flatten()
    y_test = np.load(os.path.join(path, 'y_test.npy')).flatten()
    if data_type == 'continuous':
        X_train = np.load(os.path.join(path, 'X_train.npy')).reshape(-1, 28*28)
        X_test = np.load(os.path.join(path, 'X_test.npy')).reshape(-1, 28*28)
    elif data_type == 'discrete':
        X_train = np.load(os.path.join(path, 'X_train_cat.npy')).reshape(-1, 28*28)
        X_test = np.load(os.path.join(path, 'X_test_cat.npy')).reshape(-1, 28*28)
    else:
        raise KeyError("Please make sure your keyword is 'continuous' or 'discrete' ")
    return X_train, y_train, X_test, y_test


def main(args):
    mode = args.mode
    path = args.path
    show_postirior = args.show_postirior
    show_visualization = args.show_visualization
    test_index = [int(i.strip()) for i in args.test_index.split(',')]

    if mode == 'MultinomialNB':
        X_train, y_train, X_test, y_test = load_data(path=path, data_type='discrete')
        model = MultinomialNB()
    elif mode == 'GaussianNB':
        X_train, y_train, X_test, y_test = load_data(path=path, data_type='continuous')
        model = GaussianNB()
    else:
        raise KeyError("the mode of the Naive Bayes must be 'MultinomialNB' or 'GaussianNB' ")

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print('Accuracy: %.4f\n'%metrics.accuracy_score(y_pred, y_test))

    model.make_report(X_test, y_test)
    for i in test_index:
        if show_postirior:
            model.report.sample_postirior_log(i)
        if show_visualization:
            model.report.sample_visualize(i)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--path", help="the path of the dataset", default='dataset/mnist')
    parser.add_argument("--mode", help="GaussianNB or MultinomialNB", default='MultinomialNB')
    parser.add_argument("--show_postirior", help="whether to show the postirior result by given test index", action="store_false")
    parser.add_argument("--show_visualization", help="whether to visualize the result by given test index", action="store_false")
    parser.add_argument("--test_index", help="the index of the testing set to make visualization", default='0')
    args = parser.parse_args()
    main(args)