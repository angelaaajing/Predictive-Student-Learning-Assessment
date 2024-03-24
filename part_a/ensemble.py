import numpy as np
import torch
from torch.autograd import Variable
from sklearn.impute import KNNImputer

from item_response import irt, sigmoid, evaluate as irt_evaluate
from knn import knn_impute_by_user
from neural_network import train as train_nn, AutoEncoder, evaluate as nn_evaluate
from utils import load_train_sparse, load_valid_csv, load_train_csv, sparse_matrix_predictions, load_public_test_csv, sparse_matrix_evaluate

def ensemble_evaluate(val_dic, model1_preds, model2_theta, model2_beta, model3_sparse_matrix, model3, weight):
    correct = 0
    total = 0
    for i, q in enumerate(val_dic["question_id"]):
        model1_pred = model1_preds[i] >= 0.5

        # get model 2's prediction
        u = val_dic["user_id"][i]
        x = (model2_theta[u] - model2_beta[q]).sum()
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

    # load trainng matrix
    train_matrix = load_train_sparse("../data")
    size = np.shape(train_matrix)
    train_dict = load_train_csv("../data")
    len_dict = len(train_dict["question_id"])

    # load validation matrix
    val_dic = load_valid_csv("../data")

    # bootstrap the training dict and create 3 samples
    samples = []

    for i in [1, 2, 3]:
        curr_order = sampler.integers(low=0, high=(len_dict - 1), size=len_dict)
        print(curr_order)
        curr_dict = {"question_id": [], "user_id": [], "is_correct": []}
        for j in curr_order:
            for type in ["question_id", "user_id", "is_correct"]:
                curr_dict[type].append(train_dict[type][j])
        samples.append(curr_dict)

    # model 1 using kNN, train w sample 1
        # user based w k=11 bc got highest test accuracy
    model1_sparse_matrix = np.full(size, np.nan)
    for i in range(0, len(samples[2]["user_id"])):
        row_i = samples[0]["user_id"][i]
        col_i = samples[0]["question_id"][i]
        value = samples[0]["is_correct"][i]
        model1_sparse_matrix[row_i][col_i] = value

    model1 = KNNImputer(n_neighbors=11)
    model1_mat = model1.fit_transform(model1_sparse_matrix)
    model1_preds = sparse_matrix_predictions(val_dic, model1_mat) # NOTE** thresh is .5 by default

    # model 2 using Item Response Theory, train w sample 2
    model2_theta, model2_beta, model2_acc_lst, _, _ = irt(samples[1], val_dic, 0.01, 14)

    # model 3 using Neural Networks, train w sample 3
    model3_sparse_matrix = np.full(size, np.nan)
    for i in range(0, len(curr_dict["user_id"])):
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
    train_nn(model3, lr=0.05, lamb=0.001, train_data=model3_sparse_matrix, zero_train_data=model3_zero_train_matrix, valid_data=val_dic, num_epoch=10)


    # generate 3 sets of predictions by using the base model and avg the predicted correctness
    weights_test = [[1, 1, 1], [1.5, 0.75, 0.75], [0.75, 1.5, 0.75], [0.75, 0.75, 1.5]]
    best_weight = [1, 1, 1]
    acc = []
    best_acc = 0

    for weight in weights_test:
        curr_acc = ensemble_evaluate(val_dic, model1_preds, model2_theta, model2_beta, model3_sparse_matrix, model3, weight)
        acc.append(curr_acc)
        if curr_acc > best_acc:
            best_acc = curr_acc
            best_weight = weight
        print(f"validation accuracy of weight={weight} is {curr_acc}")

    test_dic = load_public_test_csv("../data")
    model1_test_preds = sparse_matrix_predictions(test_dic, model1_mat)
    test_acc = ensemble_evaluate(test_dic, model1_test_preds, model2_theta, model2_beta, model3_sparse_matrix, model3, best_weight)
    print(f"test accuracy is {test_acc}")

    # to get training acc
    model1_train_acc = sparse_matrix_evaluate(samples[0], model1_sparse_matrix)
    model2_train_acc = irt_evaluate(samples[1], model2_theta, model2_beta)
    model3_train_acc = nn_evaluate(model3, model3_zero_train_matrix, samples[2])
    train_acc = (best_weight[0] * model1_train_acc + best_weight[1] * model2_train_acc + best_weight[2] * model3_train_acc)/3
    # print(f"train accuracy is {train_acc}")
    print(model1_train_acc)
    print(model2_train_acc)
    print(model3_train_acc)

    model3_val_acc = nn_evaluate(model3, model3_zero_train_matrix, val_dic)
    print(model3_val_acc)

    model3_test_acc = nn_evaluate(model3, model3_zero_train_matrix, test_dic)
    print(model3_test_acc)



if __name__ == "__main__":
    main()
