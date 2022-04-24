import argparse
import os
import numpy as np
from handwritten_recognition_system import Handwrittten_recognition_system
from utils.handwritten_dataset_utils import view_word
from utils.Initial_and_Transition_matrix_generator import index_to_alphabet

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--images_path',type=str, 
                        default='dataset/handwritten_alphabets/words/robotics.npy')
    parser.add_argument('--distribution_dir', type=str,
                        default='trained_model/transition_matrix_and_initial_distribution')
    parser.add_argument('--observer_path', type=str,
                        default="trained_model/observer/model_cnn.pt")

    args = parser.parse_args()
    
    images_path = args.images_path
    distribution_dir = args.distribution_dir
    transition_matrices_path = os.path.join(distribution_dir, "transition_matrix.npy")
    init_distribution_path = os.path.join(distribution_dir, "initial_distribution.npy")
    observer_path = args.observer_path

    images = np.load(images_path)

    handwritten_recogonition_predictor = Handwrittten_recognition_system(transition_matrices_path, init_distribution_path, observer_path)

    for i in range(images.shape[0]):
        handwritten_recogonition_predictor.load_image(images[i,:])
        predict_states = handwritten_recogonition_predictor.make_prediction()
        predict_letters = [index_to_alphabet[x] for x in predict_states]
        predict_vocabulary =  "".join(predict_letters)
        view_word(images[i,:], predict_vocabulary, pause_time = 0.1)

if __name__ == "__main__":
    main()