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

# This will suppress ALL FutureWarning messages
# warnings.simplefilter(action='ignore', category=RuntimeWarning)


def mean_square_error(y_true, y_pred):
    y_true = np.asmatrix(y_true).reshape(-1)
    y_pred = np.asmatrix(y_pred).reshape(-1)

    return np.square(np.subtract(y_true, y_pred)).mean()

def root_mean_square_error(y_true, y_pred):

    return mean_square_error(y_true, y_pred)**0.5

def sigmoid(x):
    # Função de ativação Sigmoid.
    # Transforma o valor de entrada para um valor entre 0 e 1.
    return 1 / (1 + np.exp(-x))

def linear(x):
    """
    Função de ativação Linear.
    A saída é exatamente igual à entrada (nenhuma transformação).
    """
    return x

def calculate_h(X, omega, b, act_fuction):
    """
    Calcula a saída de um único neurônio oculto (h_L(X)).
    Assumimos que omega é um vetor coluna e X é uma matriz de entrada.
    """
    # (X @ omega) realiza o produto escalar de cada linha de X com omega
    # O resultado é um vetor de saídas para cada amostra
    return act_fuction(X @ omega + b)

def transform_verify_numpy(array):
    if (
        (isinstance(array, pd.DataFrame)) or 
        (isinstance(array, pd.Series))
        ):
        array = array.to_numpy()

    if not isinstance(array, np.ndarray):
        raise NotImplementedError("inputs e outputs need to be pd.Dataframe or np.ndarray")
    
    return array
class MLP(BaseModel):
    '''
        Comentário sobre o Multilayer Perceptron
    '''

    def __init__(
            self, 
            hidden_neurons = [10,20,30,40],
            learning_rate = [0.1, 0.01, 0.001],
            activation = ['logistic','tanh','relu'],
            solver = 'sgd'
        ):

        super().__init__()
        
        self.hidden_neurons = hidden_neurons
        self.learning_rate = learning_rate
        self.activation = activation
        self.solver = solver

    def get_predict(self, data):

        pred_test = self.best_model.predict(data['input_test_norm'])
        pred_test_denom = DataProcessing.denorm_data(data['target_train'], pred_test)
        error_test = mean_squared_error(data['target_test'], pred_test_denom)
        self.lst_results.append(error_test)

        return {
            'pred_test': pred_test,
            'pred_test_denom': pred_test_denom,
            'error_test': error_test
        }

    def grid_search(self, data):
        
        h = []

        for h in self.hidden_neurons:
            for l in self.learning_rate:
                for a in self.activation:
                    rna = MLPRegressor(hidden_layer_sizes=(h,),learning_rate_init=l,activation=a,shuffle=False, random_state=self.current_seed, solver=self.solver)
                    
                    rna.fit(data['input_train_norm'],data['target_train_norm']) # Treina o modelo com os dados de treinamento

                    pred_valid = rna.predict(data['input_valid_norm'])
                    pred_valid_denom = DataProcessing.denorm_data(data['target_train'], pred_valid)
                    error = mean_squared_error(data['target_valid'], pred_valid_denom) 

                    self.update_best_model(rna, error, pred_valid_denom)
        
    def train(self, data):

        seeds = 10

        for sd in range(seeds):

            self.current_seed = sd

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

class SCN_III(BaseEstimator, RegressorMixin):

    def __init__(self, 
                t_max: int = 1,
                l_max: int = 10,
                r: int = 0.5,
                lambda_min: float = 0.2,
                lambda_max: float = 1.0,
                lambda_steps: int = 3,
                error_min = 0,
                hidden_act_fuction = sigmoid
            ):
       
        self.t_max = t_max
        self.l_max = l_max
        self.r = r 
        self.error_min = error_min
        self.scaler_x = None
        self.scaler_y = None
        self.lambda_max = lambda_max
        self.lambda_min = lambda_min
        self.lambda_steps = lambda_steps
        self.hidden_act_fuction = hidden_act_fuction
        

    def fit(self, X, y):
        self.weights_list = []

        self.lambda_set = np.linspace(self.lambda_min,  self.lambda_max, self.lambda_steps)

        x_train = X.copy()
        y_train = y.copy()
        x_train = transform_verify_numpy(x_train)
        y_train = transform_verify_numpy(y_train)
    
        if y_train.ndim!=1:
            raise NotImplementedError("multiple target is not implemented")
        
        self.scaler_x = MinMaxScaler(feature_range=(0, 1))
        self.scaler_x = self.scaler_x.fit(x_train)
        
        # TODO: need to be refactored if evolved to multi target
        self.scaler_y = MinMaxScaler(feature_range=(0, 1))
        self.scaler_y.fit(y_train.reshape(-1, 1))
 
        x_train_norm =  self.scaler_x.transform(x_train)
        y_train_norm =  self.scaler_y.transform(y_train.reshape(-1, 1)).flatten()

        self.error_list = []
        input_size = x_train_norm.shape[1]
        e = np.array(y_train_norm.copy())
        
        efro = 1
        r = self.r
        
        n_samples = len(x_train_norm)
        t_max = self.t_max 
        l_max = self.l_max 
        l_i = 1

        self.wstar_list = []
        self.bstar_list = []
        h_l_matrix = np.empty((n_samples, 0))

        while(l_i<=l_max and efro > self.error_min):
            gamma = []
            w = []
            bias = []
            for y in self.lambda_set:
                for k in range(0, t_max):# step 4
                    wl = np.random.uniform(low=-y, high=y, size=input_size)
                    bL = np.random.uniform(low=-y, high=y, size=1)
                    hL = calculate_h(x_train_norm, wl, bL, self.hidden_act_fuction)
                    
                    uL =  (1 - r)/(l_i + 1)
                    
                    eq28 = (np.dot(e.T, hL.T)**2 / np.dot(hL.T, hL)) -  (1 - r - uL) * e.T * e
                    
                    if np.min(eq28) > 0:
                        w.append(wl)
                        bias.append(bL)
                        gamma.append(np.sum(eq28))
                
            if len(w) > 0:
                wstar = w[np.argmax(gamma)]
                bstar = bias[np.argmax(gamma)]
                
                self.wstar_list.append(wstar)
                self.bstar_list.append(bstar)
        
                hlstar = calculate_h(x_train_norm, wstar, bstar, self.hidden_act_fuction)
                h_l_matrix = np.hstack((h_l_matrix, hlstar.reshape(-1, 1)))
            else:
                r = r + np.random.uniform(low=0, high=1 - r , size=1)[0]
                continue
        
            # Garante que T tem 2 dimensões para a multiplicação de matrizes
            T_reshaped = np.array(y_train_norm).reshape(n_samples, -1) if  np.array(y_train_norm).ndim == 1 else y_train_norm
            self.beta_star = np.dot(np.linalg.pinv(h_l_matrix), T_reshaped) 
            #self.beta_star = beta_current#.flatten().tolist() 

            el = np.dot(h_l_matrix,  self.beta_star) - T_reshaped
            e = el.copy()
            l_i = l_i + 1
            efro = np.square(el).mean()
            self.error_list.append(efro)
                    
    def predict(self, X):
        x_test = X.copy()
        x_test_norm = transform_verify_numpy(x_test)
        x_test_norm =  self.scaler_x.transform(x_test) # Aqui foi comentado Thales

        n_test_samples, _ = x_test_norm.shape
        #H_test = np.empty((N_test_samples, len(wstar_list))
        h_predict = np.empty((n_test_samples, 0))
        for i, (omega, b) in enumerate(zip(self.wstar_list, self.bstar_list)):
            h_result = calculate_h(x_test_norm, omega, b,  self.hidden_act_fuction)
            h_predict = np.hstack((h_predict, h_result.reshape(-1, 1)))

        predictions = np.dot(h_predict,  self.beta_star)
        predictions =  self.scaler_y.inverse_transform(predictions.reshape(-1, 1)).flatten()

        return predictions

class GridSCN(BaseModel):

    def __init__(self,
                l_max = [5, 10, 20, 30],
                t_max = [10],
                r = [0.5, 0.7]
                ):

        super().__init__()
         
        self.l_max = l_max
        self.t_max = t_max
        self.r = r
         

    def grid_search(self, data):

        for l in self.l_max:
            for t in self.t_max:
                for r in self.r:

                    scn = SCN_III(
                        l_max = l, 
                        t_max = t, 
                        r = r
                    )
                    
                    scn.fit(
                        data['input_train'], 
                        data['target_train']
                    )

                    pred_valid = scn.predict(
                        data['input_valid']
                        )
                    
                    error = mean_squared_error(
                        data['target_valid'], pred_valid
                    )

                    self.update_best_model(scn, error, pred_valid)

    def get_predict(self, data):

        pred_test = self.best_model.predict(data['input_test'])
        error_test = mean_squared_error(data['target_test'], pred_test)
        self.lst_results.append(error_test)

        return {
            'pred_test': pred_test,
            'error_test': error_test
        }

    def train(self, data):

        self.grid_search(data)
        predict = self.get_predict(data) 
        
        return {
            
            'lst_results': self.lst_results,
            'pred_test': predict['pred_test'],
            'error_test': predict['error_test'],
            'pred_valid': self.best_valid_preds,
            'best_rna': self.best_model,
            'best_error': self.best_error,
            'best_errors_list': self.best_errors_list
        }
class ELM(BaseEstimator, RegressorMixin):
    model = None


    def __init__(self,
                 hidden_dim=10,
                 output_dim=1,
                 solver='sigmoid'):
        '''
        Neural Network object
        '''
        self.input_dim = None
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.solver = solver
        self.U = 0
        self.V = 0
        self.S = 0
        self.H = 0
        self.alpha = 0  # for regularization
        self.W1 = None
        self.W2 = None
    # Helper function


    def sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-0.1 * x)) - 0.5


    def predict(self, x):
        '''
        Forward pass to calculate the ouput
        '''
        x = np.matrix(x)
        if self.solver == 'sigmoid':
            hidden_activation = self.sigmoid(x @ self.W1)
        elif self.solver == 'linear':
            hidden_activation = x @ self.W1
        else:
            raise RuntimeError("solver not implemented")


        y = hidden_activation @ self.W2


        return np.array(y.flatten())[0]


    def fit(self, x, y):
        '''
        Compute W2 that lead to minimal LS
        '''

        self.input_dim = x.shape[1]
        self.W1 = np.matrix(np.random.rand(self.input_dim, self.hidden_dim))
        self.W2 = np.matrix(np.random.rand(self.hidden_dim, self.output_dim))


        X = np.matrix(x)
        Y = np.matrix(np.array(y).reshape(-1, 1))


        if self.solver == 'sigmoid':
            self.H = np.matrix(self.sigmoid(X @ self.W1))
        elif self.solver == 'linear':
            self.H = np.matrix(x @ self.W1)
        else:
            raise RuntimeError("solver not implemented")


        H = cp.deepcopy(self.H)


        self.svd(H)
        iH = np.matrix(self.V) @ np.matrix(np.diag(self.S)).I @ np.matrix(self.U).T


        self.W2 = iH * Y

        del self.H
        del self.U
        del self.S
        del self.V
        self.H = self.U = self.S = self.V = None # Garante que fiquem vazios
        
        return self


    def svd(self, h):
        '''
        Compute the Singular Value Decomposition of a matrix H
        '''
        H = np.matrix(h)
        self.U, self.S, Vt = np.linalg.svd(H, full_matrices=False)
        self.V = np.matrix(Vt).T
        return np.matrix(self.U), np.matrix(self.S), np.matrix(self.V)
class GridELM(BaseModel):

    def __init__(self,
                hidden_dim = [10, 20, 50, 100]
                ):

        super().__init__()
         
        self.hidden_dim = hidden_dim

    def grid_search(self, data):

        for h in self.hidden_dim:

            elm = ELM(
                hidden_dim = h
            )
            
            elm.fit(
                data['input_train_norm'], 
                data['target_train_norm']
            )

            pred_valid = elm.predict(
                data['input_valid_norm']
                )
            
            pred_valid_denom = DataProcessing.denorm_data(data['target_train'], pred_valid)
            error = mean_squared_error(
                data['target_valid'], pred_valid_denom
            )

            self.update_best_model(elm, error, pred_valid_denom)

    def get_predict(self, data):

        pred_test = self.best_model.predict(data['input_test_norm'])
        pred_test_denom = DataProcessing.denorm_data(data['target_train'], pred_test)
        error_test = mean_squared_error(data['target_test'], pred_test_denom)
        self.lst_results.append(error_test)

        return {
            'pred_test': pred_test,
            'pred_test_denom': pred_test_denom,
            'error_test': error_test
        }
    
    def train(self, data):
        
        seeds = 10
        for sd in range(seeds):

            self.current_seed = sd

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
