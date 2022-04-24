import pandas as pd
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import tqdm 

from utils.Initial_and_Transition_matrix_generator import alphabet_to_index, index_to_alphabet

word_list = ["learning", "in", "robotics", "school", "of", "engineering", "and", "applied", "science", "university", "pennsylvania"]

def view_alphabet(flatten_img, prediction, pause_time = None):
    plt.imshow(flatten_img.reshape(28,28), cmap="gray")
    plt.title(prediction)
    if pause_time:
        plt.pause(pause_time)
    else:
        plt.show()

def view_word(flatten_word, prediction, pause_time = None):
    word = flatten_word.reshape(-1,28,28)
    img = np.zeros((28, 28*word.shape[0]))
    for i in range(word.shape[0]):
        img[:,i*28:(i+1)*28] = word[i]

    plt.imshow(img, cmap="gray")
    plt.title(prediction)
    if pause_time:
        plt.pause(0.1)
    else:
        plt.show()

def view_word_dataset(dataset_path):
    dataset = np.load(dataset_path, allow_pickle=True)
    for i in tqdm.tqdm(range(dataset.shape[0])):
        word = dataset[i,:].reshape(-1, 28, 28)
        img = np.zeros((28, 28*word.shape[0]))
        for j in range(word.shape[0]):
            img[:,j*28:(j+1)*28] = word[j]

        plt.imshow(img, cmap="gray")
        plt.title("No:{}".format(i))
        plt.pause(0.1)

def view_alphabet_dataset(dataset_path):
    dataset = np.load(dataset_path, allow_pickle=True)
    for i in tqdm.tqdm(range(dataset.shape[0])):
        img = dataset[i,1:]
        plt.imshow(img.reshape(28,28), cmap="gray")
        plt.title("No:{}".format(i))
        plt.pause(0.1)

def get_sub_dataset(dir, dataset_num=260000):

    np.random.seed(0)

    input_path = os.path.join(dir, "A_Z Handwritten Data.csv")
    original_data = pd.read_csv(input_path).to_numpy()
    
    np.random.shuffle(original_data)

    partial_data = original_data[:dataset_num]

    output_path = os.path.join(dir, "A_Z Handwritten Data small portion.npy")

    np.save(output_path, partial_data)


def split_dataset(dir, train_ratio=0.6, cv_ratio = 0.2, test_ratio = 0.2):
    
    # Loading whole dataset
    path = os.path.join(dir, "A_Z Handwritten Data small portion.npy")

    dataset = np.load(path, allow_pickle=True)

    # Split the dataset
    np.random.seed(0)
    np.random.shuffle(dataset)

    dataset_len = dataset.shape[0]
    train_dataset = dataset[:int(train_ratio*dataset_len)]
    cv_dataset = dataset[int(train_ratio*dataset_len):-int(test_ratio * dataset_len)]
    test_dataset = dataset[-int(test_ratio * dataset_len):]

    # Make directories
    train_directory = os.path.join(dir, "train_dataset")
    if not os.path.exists(train_directory):
        os.mkdir(train_directory)
    
    cv_directory = os.path.join(dir, "cross_validation_dataset")
    if not os.path.exists(cv_directory):
        os.mkdir(cv_directory)

    test_directory = os.path.join(dir, "test_dataset")
    if not os.path.exists(test_directory):
        os.mkdir(test_directory)

    # Export train dataset 
    train_dataset_path = os.path.join(train_directory, "all.npy")
    np.save(train_dataset_path, train_dataset)
    for i in range(26):
        train_dataset_certain_letter_index = (train_dataset[:,0] == i)  
        train_dataset_certain_letter_path = os.path.join(train_directory, "{}.npy".format(index_to_alphabet[i]))
        np.save(train_dataset_certain_letter_path, train_dataset[train_dataset_certain_letter_index])

    # Export cross validation dataset
    cv_dataset_path = os.path.join(cv_directory, "all.npy")
    np.save(cv_dataset_path, cv_dataset)
    for i in range(26):
        cv_dataset_certain_letter_index = (cv_dataset[:,0] == i)  
        cv_dataset_certain_letter_path = os.path.join(cv_directory, "{}.npy".format(index_to_alphabet[i]))
        np.save(cv_dataset_certain_letter_path, cv_dataset[cv_dataset_certain_letter_index])
    
    # Export test dataset 
    test_dataset_path = os.path.join(test_directory, "all.npy")
    np.save(test_dataset_path, test_dataset)
    for i in range(26):
        test_dataset_certain_letter_index = (test_dataset[:,0] == i)  
        test_dataset_certain_letter_path = os.path.join(test_directory, "{}.npy".format(index_to_alphabet[i]))
        np.save(test_dataset_certain_letter_path, test_dataset[test_dataset_certain_letter_index])


def generate_words_dataset(dir, dataset_number=100):
    test_dataset_dir_for_single_letter = os.path.join(dir, "test_dataset")

    if not os.path.exists(os.path.join(dir, "words")):
        os.mkdir(os.path.join(dir, "words"))

    np.random.seed(0)
    for word in word_list:
        
        num_letter = len(word)
    
        data = np.zeros((dataset_number, num_letter*28*28))
        
        for i in range(num_letter):
            letter = word[i]
            alphabet_dataset_path = os.path.join(os.path.join(dir, "test_dataset"), "{}.npy".format(letter))
            alphabet_dataset = np.load(alphabet_dataset_path, allow_pickle=True)
            np.random.shuffle(alphabet_dataset)
            data[:,i*28*28: (i+1)*28*28] = alphabet_dataset[:dataset_number,1:]

        output_path = os.path.join(os.path.join(dir, "words"), "{}.npy".format(word))
        np.save(output_path, data)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str,
                        default='../dataset/handwritten_alphabets/')
    args = parser.parse_args()

    dir = args.dir

    get_sub_dataset(dir)

    split_dataset(dir)

    # view_alphabet_dataset(os.path.join(dir, "train_dataset/a.npy"))

    generate_words_dataset(dir)

    view_word_dataset(os.path.join(dir, "words/learning.npy"))


if __name__ == "__main__":
    main()