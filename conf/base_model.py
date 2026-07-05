import copy
from xml.parsers.expat import model
class BaseModel:
    def __init__(self):
        self.lst_results = []
        self.best_errors_list = []
        self.best_error = float('inf')
        self.best_valid_preds = None
        self.best_model = None
        self.current_seed = None
    
    def update_best_model(self, model, error, preds_valid):
        
        if error < self.best_error:

            self.best_model = copy.deepcopy(model)
            self.best_valid_preds = preds_valid
            self.best_error = error

            if hasattr(self.best_model, "get_params"):
                params = self.best_model.get_params()
            else:
                params = self.best_model

            self.best_errors_list.append(
                {
                'erro': error, 
                'params': params
                }
            )
