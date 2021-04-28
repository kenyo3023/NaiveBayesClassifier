#encoding=utf8
import numpy as np
import pandas as pd
import struct
import os
import random
from random import sample
from sklearn import metrics

from PIL import Image
import matplotlib.pyplot as plt

class GaussianNB():
    def __init__(self, alpha=0.0001):
        self.alpha = alpha
    
    def fit(self, X, y):
        self.N = X.shape[0]
        self.classes, self.class_count = np.unique(y, return_counts=True)
        self.priors = self.class_count / self.N
        self.mu, self.sigma = [], []
        for class_ in self.classes:
            sub = X[y == class_]
            sub_mu = np.mean(sub, axis=0)
            sub_sigma = np.var(sub, axis=0) + self.alpha
            self.mu.append(sub_mu)
            self.sigma.append(sub_sigma)
        self.mu, self.sigma = np.array(self.mu), np.array(self.sigma)
    
    def gaussian_log_likelihood(self, X):
        gaussian_ll = []
        for i, prior in enumerate(self.priors):
            log_joint = np.log(prior)
            log_molecular = - 0.5 * np.sum((((X - self.mu[i,:]) ** 2) / self.sigma[i,:]), 1)
            log_denominator = - 0.5 * np.sum(np.log(2. * np.pi * self.sigma[i, :]))
            log_ll = log_joint + log_molecular + log_denominator
            gaussian_ll.append(log_ll)
        return np.array(gaussian_ll).T
    
    def normalize(self, gaussian_log_likelihood):
        gaussian_ll = gaussian_log_likelihood
        gaussian_ll_normalize = np.divide(gaussian_ll.T, np.sum(gaussian_ll, axis=1))
        return gaussian_ll_normalize.T
    
    def predict(self, X):
        prob = self.gaussian_log_likelihood(X)
        return self.classes[np.argmax(prob, axis=1)]
    
    def predict_proba(self, X):
        prob = self.gaussian_log_likelihood(X)
        return self.normalize(prob)
        
    def make_report(self, X, y_true):
        self.X = X
        self.y_true = y_true
        self.y_gll = self.gaussian_log_likelihood(self.X)
        self.y_pred = self.classes[np.argmax(self.y_gll, axis=1)]
        self.y_prob = self.normalize(self.y_gll)
        self.report = _make_report(self.X, self.y_true, self.y_pred, self.y_prob, self.classes)


class _make_report():
    def __init__(self, X, y_true, y_pred, y_prob, classes):
        self.X = X
        self.y_true = y_true
        self.y_pred = y_pred
        self.y_prob = y_prob
        self.classes = classes
        print('Acccuracy: ', metrics.accuracy_score(y_pred, y_true))
    
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