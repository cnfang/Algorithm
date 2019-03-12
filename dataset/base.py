#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Load datasets

@author: Jenny Fang
"""

from os.path import join, dirname
import pandas as pd



def load_iris(filepath='data'):
    """
        
    load iris data and return features & label
    
    Samples : 150
    Features: 4
    Labels  : 3
    
    % Parameters
    % ----------------------
    %     filepath: directory of iris.txt
    %   
    %
    % Returns
    % ----------------------
    %     {feature, label}: tuple with feature and label
    %
    """
    direc = dirname(__file__)
    path = join(direc, filepath, 'iris.txt')
    
    data = pd.read_csv(path, delimiter="\t", header=None)
    
    return data.values[:,1:5], data.values[:,0]


