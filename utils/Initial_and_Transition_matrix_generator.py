import numpy as np
import os
from numpy.linalg import norm
import argparse
import pandas as pd
import json

alphabet_list = [chr(i) for i in range(ord('a'), ord('z')+1)]
alphabet_to_index = {k:v for k,v in zip(alphabet_list,range(26))}
index_to_alphabet = {k:v for k,v in zip(range(26),alphabet_list)}

def generate_initial_distribution(data, is_based_on_frequency=False):
    """
    Calculate the probility distribution for each alphabet shown on the first place
    """

    init_distribution = np.zeros(26)

    for i in range(data.shape[0]):

        if is_based_on_frequency:
            init_distribution[alphabet_to_index[str(data['word'][i])[0]]] += data['count'][i]
        else:
            init_distribution[alphabet_to_index[str(data['word'][i])[0]]] += 1

    return init_distribution / norm(init_distribution)

def get_the_length_of_longest_words(data):

    n = 0

    for i in range(data.shape[0]):
        if len(str(data['word'][i])) > n:
            n = len(str(data['word'][i]))

    return n

def generate_transition_matrices(data, n, is_based_on_frequency=False):
    """
    Calculate the transition matrices
    """

    T = np.zeros((n-1,26,26))

    for i in range(data.shape[0]):
        word = str(data['word'][i])
        freq = data['count'][i]
        for j in range(len(word)-1):
            previous_alphabet = word[j]
            next_alphabet = word[j+1]
            # Dependend on flag calculate whether the transimatix T will be affected by the frequency of a certain word
            if is_based_on_frequency:
                T[j,alphabet_to_index[next_alphabet],alphabet_to_index[previous_alphabet]] += freq
            else:
                T[j,alphabet_to_index[next_alphabet],alphabet_to_index[previous_alphabet]] += 1
    
    # Normalize each transition matrix by column

    for i in range(T.shape[0]):
        for j in range(T.shape[2]):
            if sum(T[i,:,j]) > 0:
                T[i,:,j]/=sum(T[i,:,j])
    
    return T

def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--words_list_path',type=str, 
                        default='../dataset/words_list/unigram_freq.csv',
                        help='The path to words frequency dataset')
    parser.add_argument('--output_dir', type=str,
                        default='../trained_model/transition_matrix_and_initial_distribution')
    parser.add_argument('--is_based_on_frequency', type=bool,
                            default=False)

    args = parser.parse_args()

    input_path = args.words_list_path
    output_dir = args.output_dir
    is_based_on_frequency = args.is_based_on_frequency

    # read data

    data = pd.read_csv(input_path)

    # get the initial distribution
    initial_distribution = generate_initial_distribution(data, is_based_on_frequency)

    # length of the longest vocabulary
    max_length = get_the_length_of_longest_words(data)

    # get the transition matrix
    T = generate_transition_matrices(data, max_length, is_based_on_frequency)

    # output the infos
    json_file = os.path.join(output_dir, 'info.json')
    json_info = {"max_length":max_length, "is_based_on_frequency":is_based_on_frequency}

    with open(json_file, 'w') as f:
        json.dump(json_info, f)

    initial_distribution_file = os.path.join(output_dir, 'initial_distribution.npy')
    transition_matrix_file = os.path.join(output_dir, 'transition_matrix.npy')
    
    np.save(initial_distribution_file, initial_distribution)
    np.save(transition_matrix_file, T)

if __name__ == "__main__":
    main()
