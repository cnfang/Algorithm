#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 13 12:55:13 2018

@author: jennyfang

testing the function of ASSOM
"""
import numpy as np
from assom import ASSOM
from sklearn.datasets import load_wine
from sklearn.preprocessing import  MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score, accuracy_score
from sklearn import svm
from collections import Counter
import matplotlib.pyplot as plt


def model_learn_eval(trainData, trainLabel, testData, testLabel, classifier='KNN'):
    
    if classifier=='SVM':
        model = svm.SVC(kernel='linear', C=1)
    elif classifier == 'KNN':
        model = KNeighborsClassifier(n_neighbors=3)
    else:
        model = KNeighborsClassifier(n_neighbors=3)
        
    model.fit(trainData, trainLabel)
    predict_trainLabel = model.predict(trainData)
    predict_testLabel = model.predict(testData)
    
    f1_train = f1_score(trainLabel, predict_trainLabel)
    accuracy_train = accuracy_score(trainLabel, predict_trainLabel)
    
    f1_test = f1_score(testLabel, predict_testLabel)
    accuracy_test = accuracy_score(testLabel, predict_testLabel)
    
    return f1_train, accuracy_train, f1_test, accuracy_test


def data_balancing(Data, Label, numHiddenNode):
    
    trainData, testData, trainLabel, testLabel = train_test_split(Data, Label, test_size=0.2)
    modelscale = MinMaxScaler((0,1))
    
    trainData_normalize = modelscale.fit_transform(trainData)
    testData_normalize = modelscale.transform(testData)
    
    count = Counter(trainLabel)
    imratio = np.round(count.most_common()[0][1]/count.most_common()[1][1])
    
    majority_index = np.argwhere(trainLabel==count.most_common()[0][0])
    majority_trainData = np.squeeze(trainData_normalize[majority_index,:])
    majority_trainLabel = np.squeeze(trainLabel[majority_index])
    
    minority_index = np.argwhere(trainLabel==count.most_common()[1][0])
    minority_trainData = np.squeeze(trainData_normalize[minority_index,:])
    
    # Oversampling
    modelassom = ASSOM(numHiddenNode, numModule=imratio-1)
    modelassom.fit(minority_trainData)
    oversample_minorityData, oversample_minorityLabel = modelassom.oversampling(minority_trainData, label=count.most_common()[1][0], topN_module=imratio-1)
    
    newtrainData = np.concatenate((majority_trainData, oversample_minorityData), axis=0)
    newtrainLabel = np.concatenate((majority_trainLabel, oversample_minorityLabel), axis=0)
    
    return newtrainData, newtrainLabel, trainData_normalize, trainLabel, testData_normalize, testLabel

def main():
    
    wine = load_wine()
    wine.target[0:130]=1
    newtrainData, newtrainLabel, trainData, trainLabel, testData, testLabel = data_balancing(wine.data, wine.target, numHiddenNode=11)
    
    # before oversampling
    _,_,bf1, bacc = model_learn_eval(trainData, trainLabel, testData, testLabel, classifier='SVM')
    # after oversampling
    _,_,af1, aacc = model_learn_eval(newtrainData, newtrainLabel, testData, testLabel, classifier='SVM')
    
    
    print('Before Oversampling: accuracy =', bacc, 'F1-Score =', bf1)
    print('After Oversampling: accuracy =', aacc, 'F1-Score =', af1)
    
if __name__ == '__main__':
    main()
