"""
Adaptive Subspace Self-Organization Map

@author: Jenny Fang
"""

import numpy as np
from numpy.random import rand
from numpy.linalg import norm
from numpy.linalg import qr

class ASSOM():
    """ This code is to implement adaptive subspace self-organizing map(ASSOM)
    """
    
    def __init__(self, numHiddenNode, numModule, numEpoch=10, sigmaInit=3, sigmaDecay=0.1, etaInit=1, etaDecay=0.4, alpha=0.001):
        """
        Parameter initialization
        
        Parameters
        ----------
        numHiddenNode: number of hidden nodes in one module (less than feature number)
        numModule    : number of competitive modules 
        numEpoch     : number of training epoch
        sigmaInit    : initial value of std in gaussian function
        sigmaDecay   : decay tate of std in gaussian function
        etaInit      : initial value of learning rate
        etaDecay     : decay rate of learning rate
        alpha: 
        
        """
        self.numHiddenNode = np.int(numHiddenNode)
        self.numModule = np.int(numModule)
        self.numEpoch = np.int(numEpoch)
        
        # std of neighborhood function(gaussian)
        self.sigmaInit = sigmaInit
        self.sigmaDecay = sigmaDecay
        
        # learning rate
        self.etaInit = etaInit
        self.etaDecay = etaDecay
        
        # if difference of basis vectors between current step and previous step is too small, ignore he difference
        self.alpha = alpha
            
    def fit(self, data):
        """
        Fit the ASSOM model given training data.
        
        Parameters
        ----------
        data : array-like, shape (n_samples, n_features)
            Training data, where n_samples is the number of samples
            and n_features is the number of features.

        Returns
        -------
        self : object
            Returns the instance itself.
        
        """
        data = np.array(data, dtype=np.float64)
        self.numObservation = data.shape[0]
        self.numFeature = data.shape[1]
        
        self._fit(data)
        return self
        
    
    def _fit(self, data):
        """
        start training the ASSOM given data
        
        Parameters
        ----------
        data : array-like, shape (n_samples, n_features)
            Training data, where n_samples is the number of samples
            and n_features is the number of features.
        """
        self.som = rand(self.numModule, self.numFeature, self.numHiddenNode)-0.5
        di = np.zeros((self.numModule, self.numFeature, self.numHiddenNode), dtype=np.float64)
        re1 = rand(self.numModule, self.numFeature, self.numHiddenNode)
        pre = np.zeros((self.numModule, self.numFeature, self.numHiddenNode), dtype=np.float64)
        tenorm = np.zeros(self.numModule, dtype=np.float64)
        g = np.zeros(self.numModule, dtype=np.float64)
        ed = np.zeros((self.numFeature, self.numModule), dtype=np.float64)
        
        self.error = np.zeros((self.numEpoch, self.numModule),dtype=np.float64)
        credit = np.array(range(self.numModule,0,-1))/self.numModule
        tmpc = np.zeros(self.numModule)
        
        for epochi in range(0, self.numEpoch):
            
            sigma = self.sigmaInit * np.exp(-epochi * self.sigmaDecay)
            eta = self.etaInit * np.exp(-epochi * self.etaDecay)
           
            err_sum = 0
            
            for trainIndex in range(self.numObservation):
                trainVector = data[trainIndex,:]
                
                for modulei in range(self.numModule):
                    project = np.matmul(self.som[modulei,:,:].transpose(),trainVector)
                    ed[:,modulei] = np.matmul(project, self.som[modulei,:,:].transpose())
                    tenorm[modulei] = norm(trainVector.transpose()-ed[:,modulei])**2
                    
                winner = np.argmin(tenorm)
                err_sum = err_sum + tenorm[winner]
                
                self.error[epochi,:] = self.error[epochi,:]+tenorm.transpose()
                
                for modulei in range(self.numModule):
                    tmpc[modulei] = norm(ed[:,modulei]-ed[:,winner])
                sortindex = np.argsort(tmpc)
                
                
                for modulei in range(self.numModule):
                    g[modulei] = credit[sortindex[modulei]]
                    #g[modulei] = np.exp(-0.5*(norm(ed[:,modulei]-ed[:,winner]))/(sigma**2))
                    tmp = np.identity(self.numFeature) + eta*g[modulei]*np.matmul(trainVector.transpose(), trainVector)/(norm(ed[:,modulei])*norm(trainVector))
                    self.som[modulei,:,:] = np.matmul(tmp, self.som[modulei,:,:])     
                    print('t learning rate = ', eta*g[modulei], 'eta=', eta, 'g[modulei]=',g[modulei], 'exp=', -0.5*(norm(ed[:,modulei]-ed[:,winner]))/(sigma**2))
                    
                    di[modulei,:,:] = self.alpha*np.abs(self.som[modulei,:,:]-pre[modulei,:,:])
                    re1[modulei,:,:] = np.abs(self.som[modulei,:,:])-di[modulei,:,:]
                    mava = np.maximum(0,re1)
                    self.som[modulei,:,:] = np.sign(self.som[modulei,:,:])*mava[modulei,:,:]   
                     
                    Q,_ = qr(self.som[modulei,:,:])
                    self.som[modulei,:,:] = Q

                pre = self.som
            
            self.rmse = np.sqrt(err_sum/self.numObservation)
            print('===Iteration ', epochi, '====', end=': ')
            print('rmse' , np.round(self.rmse,3), 'error =', np.round(self.error[epochi,:],3), 'Neighbor Weight =' , np.round(g,5), 'Learning rate =', np.round(eta,3), 'sigma =', np.round(sigma,3),)
            print()
        # Finish training
        print('--------Training Phrase ends--------')
        
        sortIndex = np.argsort(tenorm)
        self.som = self.som[sortIndex,:,:]
        
        
    def transform(self, X, topN_module=1):
        """
            transform the X given the trained model 
            
            Parameters
            ----------
            X : array-like, shape (n_samples, n_features)
                Training data, where n_samples is the number of samples
                and n_features is the number of features.
            topN_module: between [1, self.numModule], return the projection in top-n module 
    
            Returns
            -------
            Xtrans:  array-like, shape (n_samples, n_feature )
                     n_features = self.numHiddenNode*numModule.
            
        """
        X = np.array(X, dtype=np.float64)
        Xtrans = np.matmul(X, self.som[0,:,:])
        if topN_module > 1:
            for i in range(1,topN_module):
                feature = np.matmul(X, self.som[i,:,:])
                Xtrans = np.concatenate((Xtrans, feature), axis=1)
        
        return Xtrans
        
        
    def oversampling(self, X, label, topN_module=1):
        """
            oversampling the data X give the trained module, all observations in data X should be in the same class
            
            Parameters
            ----------
            X : array-like, shape (n_samples, n_features)
                Training data, where n_samples is the number of samples
                and n_features is the number of features.
            topN_module: between [1, self.numModule], return the projection in top-n module 
    
            Returns
            -------
            Xoversample:  array-like, shape (n_samples, n_feature)
                         n _samples = (number of obervation in X) * topN_module,
                         and n_feature is the same dimension as X
            
        """
        X = np.array(X, dtype=np.float64)
        topN_module = np.int(topN_module)
        project = np.matmul(X, self.som[0,:,:])
        Xoversample = np.matmul(project, self.som[0,:,:].transpose())
        Xoversample = np.concatenate((X, Xoversample), axis=0)
        
        if topN_module > 1:
            for i in range(1,topN_module):
                tmpproject = np.matmul(X, self.som[i,:,:])
                datatmp = np.matmul(tmpproject, self.som[i,:,:].transpose())
                Xoversample = np.concatenate((Xoversample, datatmp), axis=0)
        
        Labeloversample = np.ones(Xoversample.shape[0])*label
        return Xoversample, Labeloversample
            
            
    
    
