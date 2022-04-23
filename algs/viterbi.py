import numpy as np
from algs.HMM import HMM

class Viterbi(HMM):
    
    def __init__(self):
        super().__init__()
    
    def calculate_delta_and_parent(self):
        delta = np.zeros((self.num_states, 26))
        
        # Initialize delta
        delta[0,:] = self.init_distribution * self.likelihood[0,:]
        parents = []
        
        for i in range(1, self.num_states):
            delta[i,:] = np.max(self.transition[i-1] * delta[i-1,:], axis=1) * self.likelihood[i,:]
            parents.append(np.argmax(self.transition[i-1] * delta[i-1,:], axis=1))

        return delta, parents
    
    def get_predict_states(self):
        
        delta, parents = self.calculate_delta_and_parent()

        predict_states = []

        predict_states.append(np.argmax(delta[-1,:]))

        while parents:
            parent = parents.pop()
            predict_states.append(parent[predict_states[-1]])

        predict_states.reverse()

        return predict_states