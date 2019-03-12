import numpy as np

def initialization_membership(numCluster, numFeature):
    
    U = np.random.rand(numCluster,numFeature)
    np.sum(U,axis=0)
    
    return U

class Fuzzy_Cmeans():
    def init(self, inputData, numCluster, numIter=50, threshold=0.01, weight=2):
        """
        % Parameters
        % ----------------------
        %     inData:      training data of size nxp, where n is observation number, and p is feature number
        %     numCluster:  number of cluster in inData
        %     iteration:   number of training iterations
        %     termination: termination threshod
        %        
        """
        self.numTrainSample = inputData.shape[0]
        self.numFeature     = inputData.shape[1]
        self.numCluster     = numCluster
        self.numIteration   = numIter
        self.threshold      = threshold
        self.weight         = weight
        
        self.centers = np.zeros(shape=[self.numCluster,self.numFeature], dtype=np.float32)
        self.memship = initialization_membership(numCluster=self.numCluster, numFeature=self.numFeature)
        
    def fit(self):
         
    def predict(self):
