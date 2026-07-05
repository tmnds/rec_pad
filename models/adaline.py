from conf.processing import DataProcessing

import numpy as np

from conf.base_model import BaseModel
from conf.processing import DataProcessing

from sklearn.metrics import mean_squared_error

class Adaline(BaseModel):
    '''
        Modelo Adaline 
    '''

    def __init__(
            self, 
            learning_rate = [0.1, 0.01, 0.001],
            epochs = [500, 1000]

        ):

        super().__init__()
        
        self.learning_rate = learning_rate
        self.epochs = epochs

    def get_predict(self, data):

        X_test = np.hstack((
            data['input_test_norm'],
            np.ones((len(data['input_test_norm']), 1))
        ))

        pred_test = X_test.dot(self.best_model)

        pred_test_denom = DataProcessing.denorm_data(data['target_train'], pred_test)
        error_test = mean_squared_error(
            data['target_test'],
            pred_test_denom
        )

        self.lst_results.append(error_test)

        return {
            'pred_test': np.asarray(pred_test),
            'pred_test_denom': pred_test_denom,
            'error_test': error_test
        }
    
    def grid_search(self, data):

        X_train = np.hstack((
            data['input_train_norm'],
            np.ones((len(data['input_train_norm']), 1))
        ))

        X_valid = np.hstack((
            data['input_valid_norm'],
            np.ones((len(data['input_valid_norm']), 1))
        ))

        y_train = data['target_train_norm']

        for e in self.epochs:
            for rates in self.learning_rate:
                W = np.random.rand(X_train.shape[1])
                for j in range(e):

                    for i in range(len(X_train)):

                        net = X_train[i].dot(W)

                        erro = y_train[i] - net

                        W = W + rates * erro * X_train[i]

                pred_valid = X_valid.dot(W)
                pred_valid_denom = DataProcessing.denorm_data(data['target_train'], pred_valid)
                error = mean_squared_error(
                    data['target_valid'],
                    pred_valid_denom
                )

                self.update_best_model(W, error, pred_valid_denom)

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