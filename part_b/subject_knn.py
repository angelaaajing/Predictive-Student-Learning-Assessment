from part_a import load_train_csv, load_valid_csv, load_public_test_csv

import csv
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

import os

# idea: for each student, make a KNN that finds the questions that they've done w the most similar subjects

# step 1: load question to subject dict
def load_q_subject_matrix(root_dir="../data"):
    path = os.path.join(root_dir, "question_meta.csv")
    if not os.path.exists(path):
        raise Exception("The specified path {} does not exist.".format(path))
    
    # Initialize the data.
    data = np.zeros((1774, 388))
    # Iterate over the row to fill in the data.
    with open(path, "r") as csv_file:
        reader = csv.reader(csv_file)
        for row in reader:
            try:
                q_id = int(row[0])
                str_list = row[1]
                subjects = str_list.strip('][').split(', ')
                for subject in subjects:
                    # say that question has that subject
                    data[q_id][int(subject)] = 1
            except ValueError:
                # Pass first row.
                pass
            except IndexError:
                # is_correct might not be available.
                pass
    return data

# make a KNN 
def make_knn_per_student_by_subject(training_data, q_subject_dicts, user_id, k):
    # make its own KNN
    subjects = []
    correctness = []

    for i, u in enumerate(training_data["user_id"]):
        if u == user_id:
            # then the point is valid
            curr_q = training_data["question_id"][i]
            curr_subjects = q_subject_dicts[curr_q]
            subjects.append(curr_subjects)

            curr_correctness = training_data["is_correct"][i]
            correctness.append(curr_correctness)
    subjects = np.array(subjects)

    classifier = KNeighborsClassifier(n_neighbors=k)
    classifier.fit(subjects, correctness)
    return classifier

def subject_knn(training_data, q_subject_dicts, i, u, k):
    model = make_knn_per_student_by_subject(training_data, q_subject_dicts, u, k)
    
    curr_q = training_data["question_id"][i]
    curr_subjects = q_subject_dicts[curr_q]

    reshaped = curr_subjects.reshape(1, -1)

    prediction = model.predict(reshaped) >= 0.5

    return prediction


def knn_acc(training_data, q_subject_dicts, valid_dict, k):
    correct = 0
    total = 0

    for i, u in enumerate(valid_dict["user_id"]):
        prediction = subject_knn(training_data, q_subject_dicts, i, u, k)

        if prediction == valid_dict["is_correct"][i]:
            correct += 1
        total += 1
    return correct / total

def main():
    training_data = load_train_csv("./data")
    q_subject_dict = load_q_subject_matrix("./data")
    valid_dict = load_valid_csv("./data")
    test_dict = load_public_test_csv("./data")

    k_list = [1, 2, 3, 5, 10, 12]
    accuracy = []
    best_k = 1
    best_acc = 0

    for k in k_list:
        curr_acc = knn_acc(training_data, q_subject_dict, valid_dict, k)
        accuracy.append(curr_acc)
        print(curr_acc)
        if curr_acc > best_acc:
            best_acc = curr_acc
            best_k = k
    test_acc = knn_acc(training_data, q_subject_dict, test_dict, best_k)
    print(f"test acc is {test_acc}")

if __name__ == "__main__":
    main()