# CPSC 371 AI - Assignment 2

# The unsupervised guys

# Sukirat Singh Dhillon 
# Karsten Ngai Nakamura
# Akshay Arulkrishnan
# Amaan Hingora

# Due: March 23, 2026

import random
import time
import matplotlib.pyplot as plt


def load_training_data(file_path):
    # reading full dataset and splitting into labels + features
    rows = []
    labels = []
    features = []

    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            row = line.split(",")
            rows.append(row)
            labels.append(row[0])       # first column = label (e/p)
            features.append(row[1:])    # rest = features

    return rows, labels, features


def load_unknown_data(file_path):
    # reading unknown mushrooms (no labels here)
    rows = []

    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            row = line.split(",")
            rows.append(row)

    return rows


def mismatch_distance(row1, row2):
    # simple distance: count mismatches (since all categorical)
    distance = 0
    for i in range(len(row1)):
        if row1[i] != row2[i]:
            distance += 1
    return distance


def stratified_split(features, labels, train_ratio=0.8, seed=42):
    # splitting data but keeping edible/poisonous balanced
    edible = []
    poisonous = []

    for x, y in zip(features, labels):
        if y == "e":
            edible.append((x, y))
        else:
            poisonous.append((x, y))

    rng = random.Random(seed)
    rng.shuffle(edible)
    rng.shuffle(poisonous)

    edible_split = int(len(edible) * train_ratio)
    poisonous_split = int(len(poisonous) * train_ratio)

    train_data = edible[:edible_split] + poisonous[:poisonous_split]
    test_data = edible[edible_split:] + poisonous[poisonous_split:]

    rng.shuffle(train_data)
    rng.shuffle(test_data)

    train_features = [x for x, y in train_data]
    train_labels = [y for x, y in train_data]

    test_features = [x for x, y in test_data]
    test_labels = [y for x, y in test_data]

    return train_features, train_labels, test_features, test_labels


def knn_predict(train_features, train_labels, test_row, k):
    # computing distances to all training points
    distances = []

    for i in range(len(train_features)):
        d = mismatch_distance(test_row, train_features[i])
        distances.append((d, train_labels[i]))

    distances.sort(key=lambda x: x[0])
    nearest = distances[:k]   # taking k closest

    edible_count = 0
    poisonous_count = 0
    edible_weight = 0.0
    poisonous_weight = 0.0

    for d, label in nearest:
        weight = 1 / (d + 1)   # slight weighting for tie-breaking

        if label == "e":
            edible_count += 1
            edible_weight += weight
        else:
            poisonous_count += 1
            poisonous_weight += weight

    edible_probability = (edible_count / k) * 100

    # normal majority vote, with weighted tie-break
    if edible_count > poisonous_count:
        prediction = "e"
    elif poisonous_count > edible_count:
        prediction = "p"
    else:
        prediction = "e" if edible_weight >= poisonous_weight else "p"

    return prediction, edible_probability


def evaluate_knn(train_features, train_labels, test_features, test_labels, k):
    # checking accuracy on validation set
    correct = 0

    for i in range(len(test_features)):
        prediction, _ = knn_predict(train_features, train_labels, test_features[i], k)
        if prediction == test_labels[i]:
            correct += 1

    return (correct / len(test_features)) * 100


def choose_best_k(train_features, train_labels, test_features, test_labels, candidate_ks):
    # trying multiple K values and picking best one
    accuracies = []
    best_k = candidate_ks[0]
    best_accuracy = -1

    for k in candidate_ks:
        accuracy = evaluate_knn(train_features, train_labels, test_features, test_labels, k)
        accuracies.append(accuracy)

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_k = k

    return best_k, best_accuracy, accuracies


def save_predictions(predictions, output_file):
    # writing final predictions to file
    with open(output_file, "w") as f:
        for label in predictions:
            f.write(label + "\n")


def save_plot(candidate_ks, accuracies, output_file):
    # simple accuracy vs K graph
    plt.figure(figsize=(8, 5))
    plt.plot(candidate_ks, accuracies, marker="o")
    plt.xlabel("K value")
    plt.ylabel("Validation Accuracy (%)")
    plt.title("KNN Validation Accuracy vs K")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()


def main():
    train_file = "MushroomData_8000.txt"
    unknown_file = "MushroomData_Unknwon_100.txt"
    prediction_file = "predictionResultKNN.txt"
    plot_file = "knn_accuracy_plot.png"

    train_ratio = 0.8
    candidate_ks = [1, 3, 5, 7, 9, 11]

    total_start = time.time()

    # loading full dataset + unknown data
    _, all_labels, all_features = load_training_data(train_file)
    unknown_rows = load_unknown_data(unknown_file)

    # split for validation
    train_features, train_labels, test_features, test_labels = stratified_split(
        all_features, all_labels, train_ratio=train_ratio, seed=42
    )

    # selecting best K
    selection_start = time.time()
    best_k, best_accuracy, accuracies = choose_best_k(
        train_features, train_labels, test_features, test_labels, candidate_ks
    )
    selection_end = time.time()

    # predicting unknown mushrooms using full dataset
    prediction_start = time.time()
    unknown_predictions = []
    unknown_probabilities = []

    for i in range(len(unknown_rows)):
        prediction, edible_probability = knn_predict(
            all_features, all_labels, unknown_rows[i], best_k
        )
        unknown_predictions.append(prediction)
        unknown_probabilities.append(edible_probability)

    prediction_end = time.time()

    # saving outputs
    save_predictions(unknown_predictions, prediction_file)
    save_plot(candidate_ks, accuracies, plot_file)

    total_end = time.time()

    edible_count = unknown_predictions.count("e")
    poisonous_count = unknown_predictions.count("p")

    print("========== KNN RESULTS ==========")
    print(f"Training / Validation split: {round(train_ratio * 100)} / {round((1 - train_ratio) * 100)}")
    print(f"Candidate K values tested: {candidate_ks}")
    print(f"Best K: {best_k}")
    print(f"Validation Accuracy: {best_accuracy:.2f}%")
    print()
    print("---------- Efficiency ----------")
    print(f"K selection + validation time: {selection_end - selection_start:.4f} seconds")
    print(f"Unknown prediction time: {prediction_end - prediction_start:.4f} seconds")
    print(f"Total runtime: {total_end - total_start:.4f} seconds")
    print(f"Average time per unknown mushroom: {(prediction_end - prediction_start) / len(unknown_rows):.6f} seconds")
    print()
    print("---------- Unknown Predictions ----------")
    for i in range(len(unknown_rows)):
        print(f"{unknown_probabilities[i]:.2f}% of unknown mushroom #{i + 1} being edible -> predicted {unknown_predictions[i]}")
    print()
    print("---------- Output Files ----------")
    print(f"Predictions written to: {prediction_file}")
    print(f"Graph saved as: {plot_file}")
    print()
    print("---------- Prediction Summary ----------")
    print(f"Edible predictions: {edible_count}")
    print(f"Poisonous predictions: {poisonous_count}")
    print("================================")


if __name__ == "__main__":
    main()