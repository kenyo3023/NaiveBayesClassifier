#encoding=utf8
import numpy as np
import pandas as pd
import struct
import os
import random
from random import sample
from sklearn import metrics
from sklearn.preprocessing import LabelBinarizer

from PIL import Image
import matplotlib.pyplot as plt

class MultinomialNB():
    def __init__(self, alpha=0.0001):
        self.alpha = alpha
    
    def fit(self, X, y):
        self.num_features = X.shape[1]
        Y, self.classes = self.onehot_transform(y)
        self.num_classes = len(self.classes)
        _, self.class_count = np.unique(Y, axis=0, return_counts=True)
        self.feature_count = np.dot(Y.T, X)
        
        self.feature_log_likelihood()
        self.class_log_likelihood()
    
    def onehot_transform(self, y):
        labelbin = LabelBinarizer()
        return labelbin.fit_transform(y), labelbin.classes_
        
    def feature_log_likelihood(self):  
        '''
            self.cf_count = count of (class x feature)
            self.c_count = count of (class)
        '''
        self.class_feature_count = self.feature_count + self.alpha
        self.sum_class_feature_count = np.sum(self.class_feature_count, axis=1)
        self.feature_log_prob_ = (np.log(self.class_feature_count) - \
                                  np.log(self.sum_class_feature_count.reshape(-1, 1)))
        
    def class_log_likelihood(self):
        self.log_class_count = np.log(self.class_count)
        self.class_log_prior_ = (self.log_class_count - np.log(self.class_count.sum()))
    
    def multinomial_log_likelihood(self, X):
        return np.dot(X, self.feature_log_prob_.T) + self.class_log_prior_
    
    def normalize(self, multinomial_log_likelihood):
        multinomial_ll = multinomial_log_likelihood
        multinomial_ll_normalize = np.divide(multinomial_ll.T, np.sum(multinomial_ll, axis=1))
        return multinomial_ll_normalize.T
    
    def predict(self, X):
        prob = self.multinomial_log_likelihood(X)
        return self.classes[np.argmax(prob, axis=1)]
    
    def predict_proba(self, X):
        prob = self.multinomial_log_likelihood(X)
        return self.normalize(prob)
    
    def make_report(self, X, y_true):
        self.X = X
        self.y_true = y_true
        self.y_mll = self.multinomial_log_likelihood(self.X)
        self.y_pred = self.classes[np.argmax(self.y_mll, axis=1)]
        self.y_prob = self.normalize(self.y_mll)
        self.report = _make_report(self.X, self.y_true, self.y_pred, self.y_prob, self.classes)


class _make_report():
    def __init__(self, X, y_true, y_pred, y_prob, classes):
        self.X = X
        self.y_true = y_true
        self.y_pred = y_pred
        self.y_prob = y_prob
        self.classes = classes
    
    def check1d(self, x):
        if x.ndim == 1:
            return x
        else:    
            raise ValueError("ndim must be 1")
            
    def sample_postirior_log(self, index):
        y_prob = self.check1d(self.y_prob[index])
        print("Postirior (in log scale):")
        for i in range(len(y_prob)):
            print("%s: %.16f"%(self.classes[i], y_prob[i]))
        y_pred = self.y_pred[index]
        y_true = self.y_true[index]
        print('Prediction: %d, Ans: %d\n'%(y_pred, y_true))
        
    def sample_visualize(self, index):
        x = self.check1d(self.X[index])
        x = np.where(x == 0, 0, 1)
        x = x.reshape(-1, 28)
        print('Imagination of numbers in Bayesian classifier:\n')
        print('%d: '%self.y_pred[index])
        for i in range(x.shape[1]):
            for j in range(x.shape[0]):
                print(x[i, j], end=' ')
            print()
        print('\n\n')