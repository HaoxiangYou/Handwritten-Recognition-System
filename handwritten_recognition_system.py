import numpy as np
from algs.viterbi import Viterbi
class Handwrittten_recognition_system():
    def __init__(self, transitions_path, init_distribution_path, alg="viterbi"):

        if alg == "viterbi":
            self.predictor = Viterbi()
    
        self.load_transitions(transitions_path)
        self.load_init_distribution(init_distribution_path)

    def load_image(self, image):
        """
        para:
            images: a numpy vector with size (n*28*28) where n represent the number of letter 
        """
        self.image = image
        self.num_states = int(image.shape[0] / (28*28))

    def load_transitions(self, path):
        self.transitions = np.load(path) 
    
    def load_init_distribution(self, path):
        self.init_distribution = np.load(path)

    def get_likelihood(self):
        # TODO get the likelihood
        self.likelihood = np.ones((self.num_states,26)) / 26

    def make_prediction(self):
        
        self.get_likelihood()

        self.predictor.set_num_states(self.num_states)
        self.predictor.load_transition(self.transitions[:self.num_states-1, :, :])
        self.predictor.load_initial_distribution(self.init_distribution)
        self.predictor.load_likelihood(self.likelihood)

        predict_states = self.predictor.get_predict_states()

        return predict_states

