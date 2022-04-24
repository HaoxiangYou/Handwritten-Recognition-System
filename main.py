import argparse
import os
import numpy as np
from handwritten_recognition_system import Handwrittten_recognition_system
from utils.handwritten_dataset_utils import view_word
from utils.Initial_and_Transition_matrix_generator import index_to_alphabet

def eval(images, path, label, alg="viterbi", is_vis=True):
    handwritten_recogonition_predictor = Handwrittten_recognition_system(path, alg)

    n = images.shape[0]

    succes_predict = 0

    for i in range(n):
        handwritten_recogonition_predictor.load_image(images[i,:])
        predict_states = handwritten_recogonition_predictor.make_prediction()
        predict_letters = [index_to_alphabet[x] for x in predict_states]
        predict_vocabulary =  "".join(predict_letters)

        if predict_vocabulary == label:
            succes_predict +=1
        
        if is_vis:
            view_word(images[i,:], predict_vocabulary, pause_time = 0.1)

    return succes_predict/n

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--images_dir',type=str, 
                        default='dataset/handwritten_alphabets/words')
    parser.add_argument('--vocabulary', type=str,
                        default='robotics')
    parser.add_argument('--distribution_dir', type=str,
                        default='trained_model/transition_matrix_and_initial_distribution')
    parser.add_argument('--observer_path', type=str,
                        default="trained_model/observer/model_cnn.pt")
    parser.add_argument('--is_vis', default=False, action='store_true')

    args = parser.parse_args()
    
    images_dir= args.images_dir
    vocabulary_label = args.vocabulary
    images_path = os.path.join(images_dir, vocabulary_label + ".npy")
    distribution_dir = args.distribution_dir
    transition_matrices_path = os.path.join(distribution_dir, "transition_matrix.npy")
    init_distribution_path = os.path.join(distribution_dir, "initial_distribution.npy")
    observer_path = args.observer_path
    is_vis = args.is_vis

    images = np.load(images_path)

    path = {"initial_distribution":init_distribution_path,
            "transition": transition_matrices_path,
            "observer":observer_path}

    algs = ["max_likelihood", "bayes_filter", "smooth", "viterbi"]

    for alg in algs:
        accuracy = eval(images, path, vocabulary_label, alg=alg, is_vis=is_vis)
        print("The accuracy of predict {} by using {} algorithmn is\n {}".format(vocabulary_label, alg, accuracy))


if __name__ == "__main__":
    main()