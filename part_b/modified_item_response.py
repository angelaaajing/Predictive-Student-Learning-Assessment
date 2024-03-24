from part_a import *

import numpy as np
import matplotlib.pyplot as plt

num_users = 542
num_questions = 1774


def sigmoid(x):
    """ Apply sigmoid function.
    """
    return np.exp(x) / (1 + np.exp(x))


def neg_log_likelihood(data, theta, beta, alpha):
    """ Compute the negative log-likelihood.

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param theta: Vector (ability parameter)
    :param beta: Vector (difficulty parameter)
    :param alpha: Vector (discrimination parameter)
    :return: float
    """
    log_lklihood = 0.0

    for i, q in enumerate(data["question_id"]):
        u = data["user_id"][i]
        c_ij = data["is_correct"][i]
        x = alpha[q] * (theta[u] - beta[q])
        p_ij = sigmoid(x)
        log_lklihood += c_ij * np.log(p_ij) + (1 - c_ij) * np.log(1 - p_ij)
    return -log_lklihood


def update_theta_beta_alpha(data, lr, theta, beta, alpha):
    """ Update theta and beta using gradient descent.

    You are using alternating gradient descent. Your update should look:
    for i in iterations ...
        theta <- new_theta
        beta <- new_beta
        alpha <- new_alpha

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param theta: Vector (ability parameter)
    :param beta: Vector (difficulty parameter)
    :param alpha: Vector (discrimination parameter)
    :return: tuple of vectors
    """
    for i, q in enumerate(data["question_id"]):
        u = data["user_id"][i]
        c_ij = data['is_correct'][i]
        x = alpha[q] * (theta[u] - beta[q])

        # Update theta, beta and alpha (maximize log-likelihood)
        theta[u] += lr * alpha[q] * (c_ij - sigmoid(x))
        beta[q] += lr * alpha[q] * (sigmoid(x) - c_ij)
        alpha[q] += lr * (theta[u] - beta[q]) * (c_ij - sigmoid(x))
    return theta, beta, alpha


def irt(data, val_data, lr, iterations):
    """ Train IRT model.

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param val_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param iterations: int
    :return: (theta, beta, alpha, val_acc_lst,  train_lld_lst, val_lld_lst)
    """
    # Initialize theta and beta.
    theta = np.zeros(num_users)
    beta = np.zeros(num_questions)
    alpha = np.ones(num_questions)

    train_acc_lst = []
    val_acc_lst = []
    train_lld_lst = []
    val_lld_lst = []

    for i in range(iterations):
        neg_lld = neg_log_likelihood(data, theta=theta, beta=beta, alpha=alpha)
        train_acc = evaluate(data=data, theta=theta, beta=beta, alpha=alpha)
        train_acc_lst.append(train_acc)
        val_acc = evaluate(data=val_data, theta=theta, beta=beta, alpha=alpha)
        val_acc_lst.append(val_acc)
        print("NLLK: {} \t Score: {}".format(neg_lld, val_acc))
        theta, beta, alpha = update_theta_beta_alpha(data, lr, theta,
                                                     beta, alpha)

        train_neg_lld = neg_log_likelihood(data, theta, beta, alpha)
        train_lld_lst.append(-train_neg_lld)
        val_neg_lld = neg_log_likelihood(val_data, theta, beta, alpha)
        val_lld_lst.append(-val_neg_lld)
    return (theta, beta, alpha, train_acc_lst, val_acc_lst,
            train_lld_lst, val_lld_lst)


def evaluate(data, theta, beta, alpha):
    """ Evaluate the model given data and return the accuracy.
    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}

    :param theta: Vector
    :param beta: Vector
    :param alpha: Vector
    :return: float
    """
    pred = []
    for i, q in enumerate(data["question_id"]):
        u = data["user_id"][i]
        x = (alpha[q] * (theta[u] - beta[q])).sum()
        p_a = sigmoid(x)
        pred.append(p_a >= 0.5)
    return np.sum((data["is_correct"] == np.array(pred))) \
        / len(data["is_correct"])


def main():
    train_data = load_train_csv("../data")
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")
    ##################################################################
    # Tune learning rate and number of iterations.
    # Report the training, validation and test accuracy.
    ##################################################################
    lr_lst = [0.01, 0.02, 0.03, 0.04, 0.05]
    iterations = 100

    best_accuracy = 0
    best_learning_rate = None
    best_iteration = None

    # Choose the best hyperparameters
    for lr in lr_lst:
        theta, beta, alpha, train_acc_lst, val_acc_lst, train_lld, val_lld = (
            irt(train_data, val_data, lr, iterations))

        # Update the best hyperparameter
        valid_accuracy = max(val_acc_lst)
        i = val_acc_lst.index(valid_accuracy)
        if valid_accuracy > best_accuracy:
            best_accuracy = valid_accuracy
            best_learning_rate = lr
            best_iteration = i

        # Plot training and validation accuracy as a function of iteration
        # for each learning rate
        plt.figure(figsize=(8, 6))
        plt.plot(range(iterations), train_acc_lst
                 , label='Training Accuracy')
        plt.plot(range(iterations), val_acc_lst
                 , label='Validation Accuracy')
        plt.xlabel('Iterations')
        plt.ylabel('Accuracy')
        plt.title(f'Training and Validation Accuracy with learning rate: {lr}')
        plt.legend()
        plt.show()

    print("Best hyperparameters:")
    print(f"learning rate is {best_learning_rate}, "
          f"number of iterations is {best_iteration}.")

    # Train the data with best hyperparameters
    (trained_theta, trained_beta, trained_alpha, train_acc_lst, val_acc_lst,
     train_lld, val_lld) = irt(train_data, val_data,
                               best_learning_rate, best_iteration)

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
    plt.show()

    # Report training, validation and test accuracy
    train_acc = evaluate(train_data, trained_theta, trained_beta, trained_alpha)
    val_acc = evaluate(val_data, trained_theta, trained_beta, trained_alpha)
    test_acc = evaluate(test_data, trained_theta, trained_beta, trained_alpha)
    print("Final Training Accuracy: {}".format(train_acc))
    print("Final Validation Accuracy: {}".format(val_acc))
    print("Final Test Accuracy: {}".format(test_acc))


if __name__ == "__main__":
    main()
