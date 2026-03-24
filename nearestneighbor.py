def load_training_data(file_path):
    train_rows = []
    train_labels = []
    train_features = []

    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            row = line.split(",")
            train_rows.append(row)
            train_labels.append(row[0])
            train_features.append(row[1:])

    return train_rows, train_labels, train_features


def load_unknown_data(file_path):
    unknown_rows = []

    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            row = line.split(",")
            unknown_rows.append(row)

    return unknown_rows

def main():
    #Section that reads the file, written by Akshay, including all the methods above
    train_file = "MushroomData_8000.txt"
    unknown_file = "MushroomData_Unknwon_100.txt"

    train_rows, train_labels, train_features = load_training_data(train_file)
    unknown_rows = load_unknown_data(unknown_file)

    #Section that calculates the cartesian distance and the probability that the unknown is edible, written by Karsten

    probabilities = []

    for unknownItems in range(len(unknown_rows)):
        distances = []

        for knownItems in range(len(train_rows)):
            distance = 0

            for field in range(0, 22):
                if unknown_rows[unknownItems][field] != train_rows[knownItems][field + 1]:
                    distance += 1

            distances.append(distance)

        lowestDistance = 9999
        lowestIndexes = []
        for i in range(len(distances)):
            if distances[i] < lowestDistance:
                lowestDistance = distances[i]
                lowestIndexes.clear()
                lowestIndexes.append(i)
            elif distances[i] == lowestDistance:
                lowestIndexes.append(i)

        probability = 0.0
        for i in range(len(lowestIndexes)):
            if train_rows[i][0] == "e":
                probability += 1

        probability /= len(lowestIndexes)
        probability *= 100
        # print(probability)

        probabilities.append(probability)

    for i in range(len(probabilities)):
        print("%.2f" % probabilities[i], end = "")
        print("% of unknown mushroom #", end= "")
        print(i+1, end= "")
        print(" being edible")

if __name__ == "__main__":
    main()