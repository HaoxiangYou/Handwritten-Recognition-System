import numpy as np
from algs.HMM import HMM
class Smooth(HMM):

    def __init__(self):
        super().__init__()
    
    def forward(self):
        alpha = np.zeros((self.num_states, 26))
        alpha[0,:] = self.init_distribution * self.likelihood[0,:]

        for i in range(1, self.num_states):
            alpha[i,:] = self.likelihood[i,:] * (self.transition[i-1] @ alpha[i-1 ,:])

        return alpha

    def backward(self):
        beta = np.zeros((self.num_states, 26))
        beta[-1,:] = np.ones_like(beta[-1,:])

        for i in range(self.num_states-2, -1, -1):
            beta[i,:] = (beta[i+1,:] * self.likelihood[i+1,:]) @ self.transition[i].T

        return beta

    def get_predict_states(self):
        
        alpha = self.forward()
        beta = self.backward()

        gamma = alpha * beta

        predict_states = np.argmax(gamma, axis=1).tolist()

        return predict_states