#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Jenny Fangsfn
"""
import sys
sys.path.append("/Users/fangjiening/Documents/Algorithm/")

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from clustering import Fuzzy_Cmeans
from dataset import load_iris

if __name__ == '__main__':
    

    """Step 1: Load data"""
    data,label = load_iris()
    # Normalization
    X_train = stats.zscore(data, axis=0);
    y_train = label
    
    
    """Step 2: Initialze & train FCM """
    model = Fuzzy_Cmeans(numCluster=3, numIter=50, threshold=0.001)
    model.fit(X_train)
    pred = model.predict(X_train)
    
    
    
    """Step 3: Visualize cluster centers by t-SNE & PCA"""
    # Observation using PCA projection
    numSample = X_train.shape[0];
    tsneData = TSNE(n_components=2).fit_transform(X_train)
    
    # c=np.sin(y_train) for mapping label between [0,1] in acceptable color range
    plt.subplot(2,2,1)
    plt.scatter(tsneData[:,0],tsneData[:,1],c=np.sin(y_train))
    plt.title('Iris Distribution on t-SNE space using true label')
    plt.xlabel('Dim 1')
    plt.ylabel('Dim 2')
    
    
    plt.subplot(2,2,3)
    plt.scatter(tsneData[:,0],tsneData[:,1],c=np.sin(pred))
    plt.title('Iris Distribution on t-SNE space using predictive label')
    plt.xlabel('Dim 1')
    plt.ylabel('Dim 2')
    
    
    
    # Observation using PCA projection
    pcaData = PCA(n_components=2).fit_transform(X_train)
    plt.subplot(2,2,2)
    plt.scatter(pcaData[:,0], pcaData[:,1], c=np.sin(y_train))
    plt.title('Iris Distribution on PCA space using true label')
    plt.xlabel('Dim 1')
    plt.ylabel('Dim 2')
    
    
    plt.subplot(2,2,4)
    plt.scatter(pcaData[:,0], pcaData[:,1], c=np.sin(pred))
    plt.title('Iris Distribution on PCA space using predictive label')
    plt.xlabel('Dim 1')
    plt.ylabel('Dim 2')
    
    plt.show()