import matplotlib.pyplot as plt
import numpy as np
from sklearn.impute import KNNImputer

from utils import *



def knn_impute_by_user(matrix, valid_data, k):
    """ Fill in the missing values using k-Nearest Neighbors based on
    student similarity. Return the accuracy on valid_data.

    See https://scikit-learn.org/stable/modules/generated/sklearn.
    impute.KNNImputer.html for details.

    :param matrix: 2D sparse matrix
    :param valid_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param k: int
    :return: float
    """
    nbrs = KNNImputer(n_neighbors=k)
    # We use NaN-Euclidean distance measure.
    mat = nbrs.fit_transform(matrix)
    acc = sparse_matrix_evaluate(valid_data, mat)
    print("Validation Accuracy: {}".format(acc))
    return acc


def knn_impute_by_item(matrix, valid_data, k):
    """ Fill in the missing values using k-Nearest Neighbors based on
    question similarity. Return the accuracy on valid_data.

    :param matrix: 2D sparse matrix
    :param valid_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param k: int
    :return: float
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    # the following is the same as by user, want to change one of the first 2 lines
    # flip someting, matrix ?!
    flip_matrix = np.swapaxes(matrix, 0, 1)

    knn = KNNImputer(n_neighbors=k)
    full_matrix = knn.fit_transform(flip_matrix)

    # flip back
    flip_full_matrix = np.swapaxes(full_matrix, 0, 1)

    acc = sparse_matrix_evaluate(valid_data, flip_full_matrix)
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return acc


def main():
    sparse_matrix = load_train_sparse("../data").toarray()
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")

    print("Sparse matrix:")
    print(sparse_matrix)
    print("Shape of sparse matrix:")
    print(sparse_matrix.shape)

    #####################################################################
    # TODO:                                                             #
    # Compute the validation accuracy for each k. Then pick k* with     #
    # the best performance and report the test accuracy with the        #
    # chosen k*.                                                        #
    #####################################################################
    # print(val_data)
    k = [1, 6, 11, 16, 21, 26]
    
    # Part A
    lst_of_user_acc = []

    for curr_k in k:
        curr_acc = knn_impute_by_user(sparse_matrix, val_data, curr_k)
        lst_of_user_acc.append(curr_acc)
        print(f"the validation accuracy for user-based kNN with k={curr_k} is {curr_acc}.")
    
    _, valid_user = plt.subplots()
    valid_user.plot(k, lst_of_user_acc, label="kNN by User")
    valid_user.set_xlabel("k")
    valid_user.set_ylabel("Validation Accuracy")
    valid_user.set_title("The Effect of k on Validation Accuracy of User-based kNN")
    # plt.show()

    # part B
    max_user_acc = max(lst_of_user_acc)
    max_user_k_index = lst_of_user_acc.index(max_user_acc)
    max_user_k = k[max_user_k_index]

    test_acc = knn_impute_by_user(sparse_matrix, test_data, max_user_k)
    print(f"the test accuracy for user-based kNN with k={max_user_k} is {test_acc}.")

    # part C
    lst_of_item_acc = []

    for curr_k in k:
        curr_acc = knn_impute_by_item(sparse_matrix, val_data, curr_k)
        lst_of_item_acc.append(curr_acc)
        print(f"the validation accuracy for item-based kNN with k={curr_k} is {curr_acc}.")
    
    _, valid_item = plt.subplots()
    valid_item.plot(k, lst_of_item_acc, label="kNN by Item")
    valid_item.set_xlabel("k")
    valid_item.set_ylabel("Validation Accuracy")
    valid_item.set_title("The Effect of k on Validation Accuracy of Item-based kNN")
    # plt.show()

    # max k
    max_item_acc = max(lst_of_item_acc)
    max_item_k_index = lst_of_item_acc.index(max_item_acc)
    max_item_k = k[max_item_k_index]

    test_acc = knn_impute_by_item(sparse_matrix, test_data, max_item_k)
    print(f"the test accuracy for item-based with k={max_item_k} is {test_acc}.")

    # combined plot
    _, combined = plt.subplots()
    combined.plot(k, lst_of_user_acc, label="kNN by User")
    combined.plot(k, lst_of_item_acc, label="kNN by Item")
    combined.set_xlabel("k")
    combined.set_ylabel("Validation Accuracy")
    combined.set_title("The Effect of k on Validation Accuracy")
    combined.legend()
    plt.show()

    return
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()
