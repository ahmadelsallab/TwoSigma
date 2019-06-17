'''
Created on Feb 9, 2017

@author: aelsalla
'''

import numpy as np

class LinearRegressionAgent(object):
    '''
    classdocs
    '''


    def __init__(self):
        '''
        Constructor
        '''
    
    def train(self, x, y):
        from sklearn.linear_model import LinearRegression
        
        self.model = LinearRegression()
        
        print(x.shape)
        print(y.shape)
        self.model.fit(np.array(x.values).reshape(-1,1),y.values)        
        return
    def predict(self, x):
        return self.model.predict(x)
    
    def rl_update(self, reward, state, action):
        return    