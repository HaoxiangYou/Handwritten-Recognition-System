import torch
import argparse
import os
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from utils.Initial_and_Transition_matrix_generator import alphabet_to_index, index_to_alphabet


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=14,
                kernel_size=5,
                stride=1,
                padding=2,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(14, 28, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        # fully connected layer, output 10 classes
        self.full_connect = nn.Linear(28 * 7 * 7, 26)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        # flatten the output of conv2 to (batch_size, 28 * 7 * 7)
        x = x.view(x.size(0), -1)
        x = self.full_connect(x)
        prob = nn.LogSoftmax(dim=1)(x)   # epsecially important rather than nn.Softmax() !!
        # if self.training:
        #     output = torch.argmax(prob, dim=1)  # return the label (batch_size,)
        # else:
        #     output = prob   # return the probability distribution (batch_size * 26)
        output = prob
        return output


def train(train_data, valid_data, dir='../trained_model/observer/'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 512
    epoch_length = 10

    model = CNN()
    loss_func = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # split the training data
    train_split = torch.split(train_data, batch_size)   # get a tuple

    model.train()

    epoch_losses = []

    for epoch in range(epoch_length):
        # split the training data
        epoch_loss = 0
        for train_batch in train_split:
            batch_size_true = len(train_batch)
            label = torch.zeros((batch_size_true, 26))
            label[torch.arange(0, batch_size_true), train_batch[:, 0]] = 1
            input = train_batch[:, 1:].reshape(batch_size_true, 1, 28, 28).float()

            # load the GPU
            label, input = label.to(device), input.to(device)

            # plt.imshow(input[8].numpy().reshape(28,28))

            # compute loss
            output = model(input)
            loss = loss_func(output, label)

            # update the model
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # record the training loss
            epoch_loss += loss.detach()
            epoch_losses.append(epoch_loss)

        # report epoch loss
        if epoch % 1 == 0:
            print(f'training {epoch}/{epoch_length}, loss {epoch_loss}')

    # save the model
    if not os.path.exists(dir):
        os.mkdir(dir)
    path = os.path.join(dir, "model_cnn.pt")
    torch.save(model.state_dict(), path)

    return model


def eval(model: CNN, valid_data, dir):
    model.eval()
    n = len(valid_data)
    label = valid_data[:, 0]
    input = valid_data[:, 1:].reshape(n, 1, 28, 28).float()
    output = model(input)
    prediction = torch.argmax(output, dim=1)
    acc = sum(label == prediction) / n
    print(f' Valid accuracy {acc}')
    return acc


def cal_likelihood(prediction):
    likelihood = np.zeros(26)
    n = prediction.shape[0]
    for i in range(n):
        likelihood[prediction[i]] += 1
    return likelihood / n

def main(retrain=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str,
                        default='../dataset/handwritten_alphabets/')
    parser.add_argument('--dir_obs', type=str,
                        default='../trained_model/observer/')
    parser.add_argument('--mode', type=str,
                        default='get_likelihood')
    args = parser.parse_args()

    dir = args.dir
    dir_obs = args.dir_obs
    mode = args.mode

    train_path = os.path.join(dir, "train_dataset/all.npy")
    valid_path = os.path.join(dir, "cross_validation_dataset/all.npy")
    test_path = os.path.join(dir, "test_dataset/all.npy")

    train_data = torch.from_numpy(np.load(train_path))
    valid_data = torch.from_numpy(np.load(valid_path))
    test_data = torch.from_numpy(np.load(test_path))

    if mode == "train":
        model = train(train_data, valid_data, dir=dir_obs)
    elif mode == "eval":
        # load current model to test
        # model_path = "../trained_model/observer/model_cnn.pt"
        model_path = os.path.join(dir_obs, "model_cnn.pt")
        model = CNN()
        model.load_state_dict(torch.load(model_path))

        eval(model, valid_data, dir=dir_obs)
    elif mode == "get_likelihood":
        
        model_path = os.path.join(dir_obs, "model_cnn.pt")
        model = CNN()
        model.load_state_dict(torch.load(model_path))
        model.eval()

        test_dir = os.path.join(dir, "cross_validation_dataset")
        likelihood = np.zeros((26,26))

        for i in range(26):
            path = os.path.join(test_dir,index_to_alphabet[i]+".npy")
            data = torch.from_numpy(np.load(path))[:,1:].reshape(-1,1,28,28).float()
            prediction = np.argmax(model(data).detach().numpy(),axis=1)
            likelihood[i,:] = cal_likelihood(prediction)
        
        output_path = os.path.join(dir_obs, "likelihood.npy")
            
        np.save(output_path ,likelihood)

if __name__ == "__main__":
    main()