"""
Fuzzy C-means Clustering

@author: Jenny Fang
"""

import numpy as np
from numpy.linalg import norm
from scipy.spatial.distance import cdist


def initialization_membership(numCluster, numSample):
    
    U = np.random.rand(numCluster,numSample)
    sumU = np.tile(np.sum(U,axis=0), reps=[numCluster,1])
    
    return U/sumU


class Fuzzy_Cmeans():
    def __init__(self, numCluster=5, numIter=50, threshold=0.01, weight=2):
        """
        % Parameters
        % ----------------------
        %     numCluster  :  number of cluster in inData
        %     numIteration:  number of training iterations
        %     threshold   :  termination threshold
        %        
        """
        self.numCluster     = numCluster
        self.numIteration   = numIter
        self.threshold      = threshold
        self.weight         = weight
        
        
    def fit(self, inputData):
        """
        start training fuzzy c-means
        
        % Parameters
        % ----------------------
        %     inputData: training data of size nxp, where n is observation number, and p is feature number
        %        
        """
        self.numFeature = inputData.shape[1]
        
        # initialization for centers and membership functions
        self.centers = np.zeros(shape=[self.numCluster,self.numFeature], dtype=np.float32)
        self.memship = initialization_membership(numCluster=self.numCluster, numSample=inputData.shape[0])
        
        # start training
        self._fit(inputData)
    
            
    def _fit(self, inputData):
        """
        % Parameters
        % ----------------------
        %     inputData: training data of size nxp, where n is observation number, and p is feature number
        %        
        """
        numSample = inputData.shape[0]
        newMemship = np.zeros(shape = [self.numCluster,numSample], dtype=np.float32)
        
        for epochi in range(self.numIteration):
        
            
            # update center
            denominator = np.sum(np.power(self.memship,self.weight),axis=1)
            for centeri in range(self.numCluster):
                numerator = np.matmul(np.power(self.memship[centeri,:],self.weight), inputData)
                self.centers[centeri,:] = numerator/denominator[centeri]
                
           
            
            # update membership function
            for samplei in range(numSample):
               tmp = np.tile(inputData[samplei,:], reps=[self.numCluster,1])-self.centers
               normtmp = np.power(norm(tmp, axis=1), -2)
               denominator = np.sum(np.power(normtmp,1/(self.weight-1)), axis=0)
               for centeri in range(self.numCluster):
                   numerator = np.power(norm(inputData[samplei,:]-self.centers[centeri,:]),-2/(self.weight-1))
                   newMemship[centeri,samplei] = numerator/denominator
           
            
            # update membership function given threshold
            delta = np.max(np.abs(newMemship-self.memship))
            print("Iteration ", str(epochi), ' Delta = ', str(delta))
            if delta <= self.threshold:
                break
            else:
                self.memship = newMemship
            
        
         
    def predict(self, inputData):
        """
        
        predict which cluster input data is given the trained centers of cluster
        
        % Parameters
        % ----------------------
        %     inputData: training data of size nxp, where n is observation number, and p is feature number
        %   
        %
        % Returns
        % ----------------------
        %     pred: matrix with size nx1, where n is number of samples in input data, showing the cluster of sample belongs to
        %
        """
        distance = cdist(inputData, self.centers)
        return np.argmin(distance, axis=1)
