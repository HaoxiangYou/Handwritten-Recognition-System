from mimetypes import init
import numpy as np
import torch
from algs.viterbi import Viterbi
from algs.smooth import Smooth
from algs.bayes_filter import BayesFilter
from algs.max_likelihood import MaxLikelihood
from observer.letter_recognition import CNN
class Handwrittten_recognition_system():
    def __init__(self, path, alg="viterbi", likelihood_choice="argmax"):

        """
        Paras:
            path: a dictionary contains all the relevant path
            alg: a string from ["viterbi", "smooth", "bayes_filter", "max_likelihood"] to select algorithm for making prediction
            likelihood: a string from ["obs", "argmax"] to select how to calculate likelihood,
                        "obs": directly use the CNN output probability as likelihood
                        "argmax": predict what is the letter, and use the prediction to find what's the likelihood for get this prediction.
                                The pre-determined likelihood matrix is 26 X 26 matrix where each row i indicate the P(Y|X_i). 
        """

        if alg == "viterbi":
            self.predictor = Viterbi()
        elif alg == "smooth":
            self.predictor = Smooth()
        elif alg == "bayes_filter":
            self.predictor = BayesFilter()
        elif alg == "max_likelihood":
            self.predictor = MaxLikelihood()
        else:
            raise ValueError("Invalid prediction algorithm")

        transitions_path = path["transition"]
        init_distribution_path = path["initial_distribution"]
        observer_path = path["observer"]

        self.load_transitions(transitions_path)
        self.load_init_distribution(init_distribution_path)

        self.observer = CNN()
        self.observer.load_state_dict(torch.load(observer_path))
        self.observer.eval()

        if likelihood_choice not in ["obs", "argmax"]:
            raise ValueError("Invalid likelihood obtained choice")

        self.likelihood_choice = likelihood_choice

        if likelihood_choice == "argmax":
            emission_path = path["emission"]
            self.emission = np.load(emission_path)

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
        self.likelihood = np.ones((self.num_states,26)) / 26

        self.observation = np.exp(self.observer(torch.from_numpy(self.image).reshape(self.num_states, 1, 28, 28).float()).detach().numpy())
        if self.likelihood_choice == "obs":
            self.likelihood = self.observation
        elif self.likelihood_choice == "argmax":
            for i in range(self.num_states):
                predict_letter = np.argmax(self.observation[i])
                self.likelihood[i, :] = self.emission[:, predict_letter]

    def make_prediction(self):
        
        self.get_likelihood()

        self.predictor.set_num_states(self.num_states)
        self.predictor.load_transition(self.transitions[:self.num_states-1, :, :])
        self.predictor.load_initial_distribution(self.init_distribution)
        self.predictor.load_likelihood(self.likelihood)

        predict_states = self.predictor.get_predict_states()

        return predict_states

