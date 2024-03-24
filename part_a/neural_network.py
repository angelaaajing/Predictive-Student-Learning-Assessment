from utils import *
from matplotlib import pyplot as plt
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data

import numpy as np
import torch

# Set the seed for PyTorch
torch.manual_seed(42)


def load_data(base_path="../data"):
    """ Load the data in PyTorch Tensor.

    :return: (zero_train_matrix, train_data, valid_data, test_data)
        WHERE:
        zero_train_matrix: 2D sparse matrix where missing entries are
        filled with 0.
        train_data: 2D sparse matrix
        valid_data: A dictionary {user_id: list,
        user_id: list, is_correct: list}
        test_data: A dictionary {user_id: list,
        user_id: list, is_correct: list}
    """
    train_matrix = load_train_sparse(base_path).toarray()
    valid_data = load_valid_csv(base_path)
    test_data = load_public_test_csv(base_path)

    zero_train_matrix = train_matrix.copy()
    # Fill in the missing entries to 0.
    zero_train_matrix[np.isnan(train_matrix)] = 0
    # Change to Float Tensor for PyTorch.
    zero_train_matrix = torch.FloatTensor(zero_train_matrix)
    train_matrix = torch.FloatTensor(train_matrix)

    return zero_train_matrix, train_matrix, valid_data, test_data


class AutoEncoder(nn.Module):
    def __init__(self, num_question, k=100):
        """ Initialize a class AutoEncoder.

        :param num_question: int
        :param k: int
        """
        super(AutoEncoder, self).__init__()

        # Define linear functions.
        self.g = nn.Linear(num_question, k)
        self.h = nn.Linear(k, num_question)

    def get_weight_norm(self):
        """ Return ||W^1||^2 + ||W^2||^2.

        :return: float
        """
        g_w_norm = torch.norm(self.g.weight, 2) ** 2
        h_w_norm = torch.norm(self.h.weight, 2) ** 2
        return g_w_norm + h_w_norm

    def forward(self, inputs):
        """ Return a forward pass given inputs.

        :param inputs: user vector.
        :return: user vector.
        """
        #####################################################################
        # TODO:                                                             #
        # Implement the function as described in the docstring.             #
        # Use sigmoid activations for f and g.                              #
        #####################################################################
        out = F.sigmoid(self.h(F.sigmoid(self.g(inputs))))
        #####################################################################
        #                       END OF YOUR CODE                            #
        #####################################################################
        return out


def train(model, lr, lamb, train_data, zero_train_data, valid_data, num_epoch):
    """ Train the neural network, where the objective also includes
    a regularizer.

    :param model: Module
    :param lr: float
    :param lamb: float
    :param train_data: 2D FloatTensor
    :param zero_train_data: 2D FloatTensor
    :param valid_data: Dict
    :param num_epoch: int
    :return: None
    """
    # TODO: Add a regularizer to the cost function.

    # Tell PyTorch you are training the model.
    model.train()

    # Define optimizers and loss function.
    optimizer = optim.SGD(model.parameters(), lr=lr)
    num_student = train_data.shape[0]

    train_loss_lst = []
    valid_loss_lst = []

    for epoch in range(0, num_epoch):
        train_loss = 0.

        for user_id in range(num_student):
            inputs = Variable(zero_train_data[user_id]).unsqueeze(0)
            target = inputs.clone()

            optimizer.zero_grad()
            output = model(inputs)

            # Mask the target to only compute the gradient of valid entries.
            nan_mask = np.isnan(train_data[user_id].unsqueeze(0).numpy())
            target[0][nan_mask] = output[0][nan_mask]

            # Train with L2 regularization
            loss = (torch.sum((output - target) ** 2.)
                    + (lamb / 2) * model.get_weight_norm())
            loss.backward()

            train_loss += loss.item()
            optimizer.step()

        valid_acc = evaluate(model, zero_train_data, valid_data)
        print("Epoch: {} \tTraining Cost: {:.6f}\t "
              "Valid Acc: {}".format(epoch, train_loss, valid_acc))

        # Evaluate validation loss after each epoch
        model.eval()    # Set the model to evaluation mode.
        valid_loss = 0.
        for u in valid_data["user_id"]:
            inputs = Variable(zero_train_data[u]).unsqueeze(0)
            target = inputs.clone()
            output = model(inputs)

            nan_mask = np.isnan(train_data[u].unsqueeze(0).numpy())
            target[0][nan_mask] = output[0][nan_mask]

            loss = (torch.sum((output - target) ** 2.)
                    + (lamb / 2) * model.get_weight_norm())
            loss.backward()

            valid_loss += loss.item()

        train_loss_lst.append(train_loss)
        valid_loss_lst.append(valid_loss)
        
        model.train()  # Set the model to train mode.

    return train_loss_lst, valid_loss_lst
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


def evaluate(model, train_data, valid_data):
    """ Evaluate the valid_data on the current model.

    :param model: Module
    :param train_data: 2D FloatTensor
    :param valid_data: A dictionary {user_id: list,
    question_id: list, is_correct: list}
    :return: float
    """
    # Tell PyTorch you are evaluating the model.
    model.eval()

    total = 0
    correct = 0

    for i, u in enumerate(valid_data["user_id"]):
        inputs = Variable(train_data[u]).unsqueeze(0)
        output = model(inputs)

        guess = output[0][valid_data["question_id"][i]].item() >= 0.5
        if guess == valid_data["is_correct"][i]:
            correct += 1
        total += 1
    return correct / float(total)


def main():
    zero_train_matrix, train_matrix, valid_data, test_data = load_data()

    #####################################################################
    # TODO:                                                             #
    # Try out 5 different k and select the best k using the             #
    # validation set.                                                   #
    #####################################################################
    # Tune model hyperparameters.
    k_values = [10, 50, 100, 200, 500]
    lr_lst = [0.01, 0.05, 0.1]
    num_epoch_lst = [10, 20, 30]

    best_k = None
    best_lr = None
    best_num_epoch = None
    best_valid_acc = 0

    for k in k_values:
        for lr in lr_lst:
            for num_epoch in num_epoch_lst:
                model = AutoEncoder(num_question=1774, k=k)

                # Train the model
                train(model, lr, 0, train_matrix, zero_train_matrix,
                      valid_data, num_epoch)

                # Evaluate on validation set
                valid_acc = evaluate(model, zero_train_matrix, valid_data)

                # Check if current k gives better accuracy
                if valid_acc > best_valid_acc:
                    best_valid_acc = valid_acc
                    best_k = k
                    best_lr = lr
                    best_num_epoch = num_epoch

    print("Best k:", best_k)
    print("Best learning rate:", best_lr)
    print("Best num of epoch:", best_num_epoch)

    # Train an origin model with the best hyperparameters.
    origin_model = AutoEncoder(num_question=1774, k=best_k)
    train_loss, valid_loss = train(origin_model, best_lr, 0,
                                   train_matrix, zero_train_matrix,
                                   valid_data, best_num_epoch)

    print("Validation Accuracy for origin model:"
          , evaluate(origin_model, zero_train_matrix, valid_data))

    # Plotting
    plt.plot(range(best_num_epoch), train_loss
             , label='Training Loss for origin model')
    plt.plot(range(best_num_epoch), valid_loss
             , label='Validation Loss for origin model')
    plt.xlabel('Number of Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Changes as a Function of Epoch')
    plt.legend()
    plt.show()

    # Evaluate on the test set
    test_accuracy = evaluate(origin_model, zero_train_matrix, test_data)
    print("Test Accuracy for origin model:", test_accuracy)

    # Tune the regularization penalty using best hyperparameters
    lamb_lst = [0.001, 0.01, 0.1, 1]
    best_lamb = None
    best_valid_acc = 0

    for lamb in lamb_lst:
        model = AutoEncoder(num_question=1774, k=best_k)

        # Train the model
        train(model, best_lr, lamb, train_matrix, zero_train_matrix,
              valid_data, best_num_epoch)

        # Evaluate on validation set
        valid_acc = evaluate(model, zero_train_matrix, valid_data)

        # Check if current k gives better accuracy
        if valid_acc > best_valid_acc:
            best_valid_acc = valid_acc
            best_lamb = lamb

    print("Best lambda:", best_lamb)

    # Train a final model with the best hyperparameters.
    final_model = AutoEncoder(num_question=1774, k=best_k)
    train(final_model, best_lr, best_lamb, train_matrix,
          zero_train_matrix, valid_data, best_num_epoch)

    print("Validation Accuracy for final model:",
          evaluate(final_model, zero_train_matrix, valid_data))

    # Evaluate on the test set
    test_accuracy = evaluate(final_model, zero_train_matrix, test_data)
    print("Test Accuracy for final model:", test_accuracy)
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()
