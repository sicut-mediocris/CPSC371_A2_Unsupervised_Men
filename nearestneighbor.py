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
    train_file = "MushroomData_8000.txt"
    unknown_file = "MushroomData_Unknwon_100.txt"

    train_rows, train_labels, train_features = load_training_data(train_file)
    unknown_rows = load_unknown_data(unknown_file)

    print("Train rows:", len(train_rows))
    print("Unknown rows:", len(unknown_rows))


if __name__ == "__main__":
    main()