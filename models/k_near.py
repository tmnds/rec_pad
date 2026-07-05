from conf.processing import DataProcessing
import warnings

import numpy as np
import pandas as pd
import copy as cp

from conf.base_model import BaseModel
from conf.processing import DataProcessing

from sklearn.metrics import mean_squared_error
from sklearn.neural_network import MLPRegressor
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.preprocessing import MinMaxScaler

class KNEAR(BaseModel):
    '''
        Modelo KNN 
    '''

    def __init__(
            self, 
            K = [3,5,7,11]
        ):

        super().__init__()
        
        self.K = K

    def get_predict(self, data):

        pred_idx = []

        for i in range( len(data['input_test_norm']) ):
            _, indexes = self.getNeighbors(
                data['input_test_norm'][i,:], 
                data['input_train_norm'], self.best_model
            )
        
            pred_idx.append(np.mean(data['target_train_norm'][indexes]))

        pred_test_denom = DataProcessing.denorm_data(data['target_train'], pred_idx)
        error_test = mean_squared_error(data['target_test'], pred_test_denom) 
        self.lst_results.append(error_test)

        return {
            'pred_test': np.asarray(pred_idx),
            'pred_test_denom': pred_test_denom,
            'error_test': error_test
        }
    
    
    def getNeighbors(self, X_valid_i, X_train_N, K):

        dists=[]

        for i in range(len(X_train_N)):

            dists.append(np.linalg.norm(X_valid_i-X_train_N[i,:]))
        
        indexes = np.argsort(dists)
        neighbors=X_train_N[indexes[0:K],:]
        
        return neighbors,indexes[0:K]

    def grid_search(self, data):

        for k in self.K:

            pred_idx = []

            for i in range( len(data['input_valid_norm']) ):
                _, indexes = self.getNeighbors(
                    data['input_valid_norm'][i,:], 
                    data['input_train_norm'], k
                )
            
                pred_idx.append(np.mean(data['target_train_norm'][indexes]))
            
            pred_valid_denom = DataProcessing.denorm_data(data['target_train'], pred_idx)
            error = mean_squared_error(data['target_valid'], pred_valid_denom)

            self.update_best_model(k, error, pred_valid_denom)
        
    def train(self, data):

        self.grid_search(data)   
        predict = self.get_predict(data) 

        return {
            
            'lst_results': self.lst_results,
            'pred_test': predict['pred_test'],
            'pred_test_denom': predict['pred_test_denom'],
            'pred_valid': self.best_valid_preds,
            'best_rna': self.best_model,
            'best_error': self.best_error,
            'best_errors_list': self.best_errors_list
        }