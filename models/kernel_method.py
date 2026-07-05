from sklearn.metrics import mean_squared_error
from sklearn.svm import SVR
from conf.base_model import BaseModel
from conf.processing import DataProcessing

class SVM(BaseModel):
    '''
        Comentário sobre o Support Vector Machine
    
    '''
    def __init__(self, 
                C_values=[0.1, 1, 10, 100], # Penalidade por erro
                epsilon_values = [0.01, 0.05, 0.1],
                gamma_values = ['scale', 0.01, 0.1]
                ):

        super().__init__()
        
        self.C_values = C_values
        self.epsilon_values = epsilon_values
        self.gamma_values = gamma_values

    def get_predict(self, data):
        
        pred_test = self.best_model.predict(data['input_test_norm'])
        error_test = mean_squared_error(data['target_test'], pred_test)
        self.lst_results.append(error_test)

        return {
            'pred_test': pred_test,
            'error_test': error_test
        }

    def grid_search(self, data):

        for c in self.C_values:
            for e in self.epsilon_values:
                for g in self.gamma_values:

                    svr = SVR(kernel='rbf', C=c, epsilon=e, gamma=g)
                    svr.fit(data['input_train_norm'], data['target_train'])
                
                    pred_valid = svr.predict(data['input_valid_norm'])
                    error = mean_squared_error(data['target_valid'], pred_valid) 

                    self.update_best_model(svr, error, pred_valid)
    
    def train(self, data):

        self.grid_search(data)   
        predict = self.get_predict(data) 

        return {
            
            'lst_results': self.lst_results,
            'pred_test': predict['pred_test'],
            'pred_valid': self.best_valid_preds,
            'best_rna': self.best_model,
            'best_error': self.best_error,
            'best_errors_list': self.best_errors_list
        }
