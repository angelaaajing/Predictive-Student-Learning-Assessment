from utils import *

import numpy as np
import matplotlib.pyplot as plt
import math

num_users = 542
num_questions = 1774


def sigmoid(x):
    """ Apply sigmoid function.
    """
    return np.exp(x) / (1 + np.exp(x))


def neg_log_likelihood(data, theta, beta):
    """ Compute the negative log-likelihood.

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param theta: Vector
    :param beta: Vector
    :return: float
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    log_lklihood = 0.0
    for i, q in enumerate(data["question_id"]):
        u = data["user_id"][i]
        c_ij = data["is_correct"][i]
        x = theta[u] - beta[q]
        p_ij = sigmoid(x)
        log_lklihood += c_ij * np.log(p_ij) + (1 - c_ij) * np.log(1 - p_ij)
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return -log_lklihood


def update_theta_beta(data, lr, theta, beta):
    """ Update theta and beta using gradient descent.

    You are using alternating gradient descent. Your update should look:
    for i in iterations ...
        theta <- new_theta
        beta <- new_beta

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param theta: Vector
    :param beta: Vector
    :return: tuple of vectors
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    for i, q in enumerate(data["question_id"]):
        u = data["user_id"][i]
        c_ij = data['is_correct'][i]
        x = theta[u] - beta[q]

        # Update theta and beta (maximize log-likelihood)
        theta[u] += lr * (c_ij - sigmoid(x))
        beta[q] += lr * (sigmoid(x) - c_ij)
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return theta, beta


def irt(data, val_data, lr, iterations):
    """ Train IRT model.

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param val_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param iterations: int
    :return: (theta, beta, val_acc_lst)
    """
    # TODO: Initialize theta and beta.
    theta = np.zeros(num_users)
    beta = np.zeros(num_questions)

    val_acc_lst = []
    train_lld_lst = []
    val_lld_lst = []

    for i in range(iterations):
        neg_lld = neg_log_likelihood(data, theta=theta, beta=beta)
        score = evaluate(data=val_data, theta=theta, beta=beta)
        val_acc_lst.append(score)
        print("NLLK: {} \t Score: {}".format(neg_lld, score))
        theta, beta = update_theta_beta(data, lr, theta, beta)

        train_neg_lld = neg_log_likelihood(data, theta, beta)
        train_lld_lst.append(- train_neg_lld)
        val_neg_lld = neg_log_likelihood(val_data, theta, beta)
        val_lld_lst.append(- val_neg_lld)

    # TODO: You may change the return values to achieve what you want.
    return theta, beta, val_acc_lst, train_lld_lst, val_lld_lst


def evaluate(data, theta, beta):
    """ Evaluate the model given data and return the accuracy.
    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}

    :param theta: Vector
    :param beta: Vector
    :return: float
    """
    pred = []
    for i, q in enumerate(data["question_id"]):
        u = data["user_id"][i]
        x = (theta[u] - beta[q]).sum()
        p_a = sigmoid(x)
        pred.append(p_a >= 0.5)
    return np.sum((data["is_correct"] == np.array(pred))) \
        / len(data["is_correct"])


def main():
    train_data = load_train_csv("../data")
    # You may optionally use the sparse matrix.
    sparse_matrix = load_train_sparse("../data")
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")

    #####################################################################
    # TODO:                                                             #
    # Tune learning rate and number of iterations. With the implemented #
    # code, report the validation and test accuracy.                    #
    #####################################################################
    # Tune learning rate and number of iterations
    lr_lst = [0.01, 0.05, 0.1]
    iterations = 100

    best_accuracy = 0
    best_learning_rate = None
    best_iteration = None

    # Choose the best hyperparameters
    for lr in lr_lst:
        theta, beta, val_acc_lst, _, _ = irt(train_data, val_data,
                                             lr, iterations)

        # Update the best hyperparameter
        valid_accuracy = max(val_acc_lst)
        i = val_acc_lst.index(valid_accuracy)
        if valid_accuracy > best_accuracy:
            best_accuracy = valid_accuracy
            best_learning_rate = lr
            best_iteration = i

    print("Best hyperparameters:")
    print(f"learning rate is {best_learning_rate}, "
          f"number of iterations is {best_iteration}.")

    # Train the data with best hyperparameters
    trained_theta, trained_beta, val_acc_lst, train_lld, val_lld = irt(
        train_data, val_data, best_learning_rate, best_iteration)

    # Plot training curve
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, best_iteration + 1), train_lld,
             label='Training Log-Likelihood')
    plt.plot(range(1, best_iteration + 1), val_lld,
             label='Validation Log-Likelihood')
    plt.xlabel('Iterations')
    plt.ylabel('Log-Likelihood')
    plt.title('Training and Validation Log-Likelihood')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Report validation and test accuracy
    val_acc = evaluate(val_data, trained_theta, trained_beta)
    test_acc = evaluate(test_data, trained_theta, trained_beta)
    print("Final Validation Accuracy: {}".format(val_acc))
    print("Final Test Accuracy: {}".format(test_acc))
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################

    #####################################################################
    # TODO:                                                             #
    # Implement part (d)                                                #
    #####################################################################
    # Select three questions
    j_1, j_2, j_3 = 1, 2, 3
    selected_questions = [j_1, j_2, j_3]

    min_theta_range = math.floor(min(trained_theta))
    max_theta_range = math.ceil(max(trained_theta)) + 1
    theta_value = np.arange(min_theta_range, max_theta_range, 0.1)
    plt.figure(figsize=(8, 6))
    for q in selected_questions:
        probs = [sigmoid(theta - trained_beta[q]) for theta in theta_value]
        plt.plot(theta_value, probs, label="Question {}".format(q))

    plt.xlabel("Theta Values")
    plt.ylabel("Probability of Correct Response (p(c_ij=1))")
    plt.title("The Probability of Correct Response as a Function of Theta")
    plt.legend()
    plt.show()
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()
