import numpy as np
import torch
from torch.autograd import Variable
from sklearn.impute import KNNImputer

from modified_item_response import irt, sigmoid, evaluate as irt_evaluate
from subject_knn import load_q_subject_matrix, knn_acc, subject_knn
from part_a.neural_network import train as train_nn, AutoEncoder, evaluate as nn_evaluate
from part_a import load_train_sparse, load_valid_csv, load_train_csv, load_public_test_csv, sparse_matrix_evaluate

def ensemble_evaluate(train_dict, val_dic, q_subject_dicts, k, model2_theta, model2_beta, model2_alpha, model3_sparse_matrix, model3, weight):
    correct = 0
    total = 0
    for i, q in enumerate(val_dic["question_id"]):
        u = val_dic["user_id"][i]

        model1_pred = subject_knn(train_dict, q_subject_dicts, i, u, k) >= 0.5

        # get model 2's prediction
        x = (model2_alpha[q] * (model2_theta[u] - model2_beta[q])).sum()
        model2_pred = sigmoid(x) >= 0.5

        # get model 3's prediction
        inputs = Variable(model3_sparse_matrix[u]).unsqueeze(0)
        output = model3(inputs)
        model3_pred = output[0][val_dic["question_id"][i]].item() >= 0.5

        final_pred = ((weight[0] * model1_pred + weight[1] * model2_pred + weight[2] * model3_pred)/3) >= 0.5

        if final_pred == val_dic["is_correct"][i]:
            correct += 1
        total += 1
    curr_acc = correct / total
    return curr_acc

def main():
    # set seed
    sampler = np.random.default_rng(seed=207)

    # load info
    sparse_matrix = load_train_sparse("./data")
    size = sparse_matrix.shape
    train_dict = load_train_csv("./data")
    len_dict = len(train_dict["question_id"])
    q_subject_dict = load_q_subject_matrix("./data")
    valid_dict = load_valid_csv("./data")
    test_dict = load_public_test_csv("./data")

    # bootstrap the training dict and create 3 samples
    samples = []

    for i in [1, 2, 3]:
        curr_order = sampler.integers(low=0, high=(len_dict - 1), size=len_dict)
        curr_dict = {"question_id": [], "user_id": [], "is_correct": []}
        for j in curr_order:
            for type in ["question_id", "user_id", "is_correct"]:
                curr_dict[type].append(train_dict[type][j])
        samples.append(curr_dict)

    # model 1 using kNN, train w sample 1
        # user based w k=11 bc got highest test accuracy
    # will do in evaluate section

    # model 2 using Item Response Theory, train w sample 2
    model2_theta, model2_beta, model2_alpha, model2_train_acc_lst, _, _, _= irt(samples[1], valid_dict, 0.04, 4)

    # model 3 using Neural Networks, train w sample 3
    model3_sparse_matrix = np.full(size, np.nan)
    for i in range(0, len(samples[2]["user_id"])):
        row_i = samples[2]["user_id"][i]
        col_i = samples[2]["question_id"][i]
        value = samples[2]["is_correct"][i]
        model3_sparse_matrix[row_i][col_i] = value

    model3_zero_train_matrix = model3_sparse_matrix.copy()
    # Fill in the missing entries to 0.
    model3_zero_train_matrix[np.isnan(model3_sparse_matrix)] = 0
    # Change to Float Tensor for PyTorch.
    model3_zero_train_matrix = torch.FloatTensor(model3_zero_train_matrix)
    model3_sparse_matrix = torch.FloatTensor(model3_sparse_matrix)



    model3 = AutoEncoder(num_question=1774, k=50)
    train_nn(model3, lr=0.05, lamb=0.001, train_data=model3_sparse_matrix, zero_train_data=model3_zero_train_matrix, valid_data=valid_dict, num_epoch=10)


    # generate 3 sets of predictions by using the base model and avg the predicted correctness
    weights_test = [[1, 1, 1], [1.5, 0.75, 0.75], [0.75, 1.5, 0.75], [0.75, 0.75, 1.5]]
    best_weight = [1, 1, 1]
    acc = []
    best_acc = 0

    for weight in weights_test:
        curr_acc = ensemble_evaluate(samples[0], valid_dict, q_subject_dict, 5, model2_theta, model2_beta, model2_alpha, model3_sparse_matrix, model3, weight)
        acc.append(curr_acc)
        if curr_acc > best_acc:
            best_acc = curr_acc
            best_weight = weight
        print(f"validation accuracy of weight={weight} is {curr_acc}")

    test_acc = ensemble_evaluate(train_dict, test_dict, q_subject_dict, 5, model2_theta, model2_beta, model2_alpha, model3_sparse_matrix, model3, best_weight)
    print(f"test accuracy is {test_acc}")

    # to get training acc
    model1_train_acc = knn_acc(samples[0], q_subject_dict, valid_dict, 7)
    model2_train_acc = irt_evaluate(samples[1], model2_theta, model2_beta, model2_alpha)
    model3_train_acc = nn_evaluate(model3, model3_zero_train_matrix, samples[2])
    # train_acc = (best_weight[0] * model1_train_acc + best_weight[1] * model2_train_acc + best_weight[2] * model3_train_acc)/3
    # print(f"train accuracy is {train_acc}")
    print(model1_train_acc)
    print(model2_train_acc)
    print(model3_train_acc)




if __name__ == "__main__":
    main()
