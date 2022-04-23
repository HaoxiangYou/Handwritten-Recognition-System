import numpy as np
from algs.HMM import HMM

class MaxLikelihood(HMM):
    def __init__(self):
        super().__init__()
    
    def get_predict_states(self):
        predict_states = np.argmax(self.likelihood, axis=1).tolist()
        return predict_states