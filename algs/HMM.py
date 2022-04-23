import numpy as np

class HMM:
    def __init__(self,):
        pass
    def set_num_states(self, num_states):
        """
        Parameters:
            num_states: a number n indicate how many letter the vacabulary has
        """
        self.num_states = num_states

    def load_likelihood(self, likelihood):
        """
        Parameters:
            likelihood: a n x 26 numpy matrix, each row indicate the likelihood of single letter being certain alphabet
        """
        self.likelihood = likelihood
    
    def load_transition(self, transition):
        """
        Parameters:
            transitions: a n-1 X 26 X 26 numpy array containing n transition matrices, each 26(Next) X 26(Current) indicate a transtion matrix, from current state to next state.
        """
        self.transition = transition
    
    def load_initial_distribution(self, initial_distribution):
        """
        Parameters:
            initial_distribution: a 26X1 numpy vector indicate the initial distribution of a letter begin first among whole frequent use vacabulary
        """
        self.init_distribution = initial_distribution

    def get_predict_states(self):
        raise NotImplementedError