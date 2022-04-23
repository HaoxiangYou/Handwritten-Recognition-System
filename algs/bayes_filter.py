import numpy as np
from algs.HMM import HMM
class BayesFilter(HMM):

    def __init__(self):
        super().__init__()

    def get_predict_states(self):
        state_prob = np.zeros((self.num_states, 26))
        state_prob[0,:] = self.init_distribution * self.likelihood[0,:]

        for i in range(1, self.num_states):
            state_prob[i,:] = self.likelihood[i,:] * (self.transition[i-1] @ state_prob[i-1,:])

        predict_states = np.argmax(state_prob, axis=1).tolist()

        return predict_states