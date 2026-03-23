"""
Mushroom Classification using Perceptron
CPSC 371 - Assignment 2

Sukirat Singh Dhillon,230155722
Karsten Ngai Nakamura
Amaan Hingora
Akshay ArulKrishnan

All categorical attributes are one-hot encoded. Missing values ('?') get
their own indicator column (treated as an additional category). The perceptron
uses a step activation function: output = 1 if dot(w, x) + bias >= 0 else 0.
Labels: 'e' -> 1 (edible), 'p' -> 0 (poisonous).
"""

import numpy as np
import random

# Hyperparameters
LEARNING_RATE = 0.01
EPOCHS        = 100
TRAIN_RATIO   = 0.80   # 80% train, 20% validation
RANDOM_SEED   = 42


random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# 1. Load raw data

def load_training(path):
    """Read training file. Returns list of (label_str, [attr_str, ...]) tuples."""
    samples = []
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(',')
            label = parts[0]          # 'e' or 'p'
            attrs = parts[1:]         # 22 categorical attributes
            samples.append((label, attrs))
    return samples


def load_unknown(path):
    """Read unknown test file. Returns list of [attr_str, ...] lists."""
    samples = []
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            attrs = line.split(',')   # 22 attributes, no label
            samples.append(attrs)
    return samples


# 2. Build vocabulary (fit on training data only)

def build_vocab(samples):
    """
    For each of the 22 attribute positions, collect all unique values
    seen in the training set. '?' is kept as a valid category.
    Returns a list of 22 dicts: {value -> one_hot_index}.
    """
    n_attrs = len(samples[0][1])
    vocab = [{} for _ in range(n_attrs)]
    for _, attrs in samples:
        for col, val in enumerate(attrs):
            if val not in vocab[col]:
                vocab[col][val] = len(vocab[col])
    return vocab


def encode_row(attrs, vocab):
    """
    One-hot encode a single row of attributes using the pre-built vocab.
    Unknown values (unseen during training) are mapped to a zero vector
    for that column (effectively ignored).
    """
    vec = []
    for col, val in enumerate(attrs):
        n_cats = len(vocab[col])
        one_hot = [0] * n_cats
        if val in vocab[col]:
            one_hot[vocab[col][val]] = 1
        vec.extend(one_hot)
    return vec


def encode_dataset(samples, vocab, has_label=True):
    """
    Encode all samples. Returns (X, y) where X is a 2-D numpy array
    and y is a 1-D numpy array of ints (1=edible, 0=poisonous).
    If has_label=False, y is None.
    """
    X, y = [], []
    for item in samples:
        if has_label:
            label, attrs = item
            X.append(encode_row(attrs, vocab))
            y.append(1 if label == 'e' else 0)
        else:
            X.append(encode_row(item, vocab))
    X = np.array(X, dtype=np.float64)
    y = np.array(y, dtype=np.float64) if has_label else None
    return X, y


# 3. Train / validation split

def train_val_split(X, y, train_ratio, seed=42):
    """Shuffle and split into training and validation sets."""
    n = len(y)
    indices = list(range(n))
    random.Random(seed).shuffle(indices)
    split = int(n * train_ratio)
    train_idx = indices[:split]
    val_idx   = indices[split:]
    return X[train_idx], y[train_idx], X[val_idx], y[val_idx]


# 4. Perceptron

class Perceptron:
    """
    Classic single-layer perceptron with step activation.
    Weights are initialized to small random values; bias starts at 0.
    Update rule (on misclassification):
        w = w + lr * (target - prediction) * x
        b = b + lr * (target - prediction)
    """

    def __init__(self, n_features, learning_rate=0.01):
        # Small random weight initialization helps break symmetry
        self.weights = np.random.uniform(-0.05, 0.05, n_features)
        self.bias    = 0.0
        self.lr      = learning_rate

    def _activate(self, z):
        """Step function: fire (1) if net input >= 0, else silence (0)."""
        return 1 if z >= 0 else 0

    def predict_one(self, x):
        """Compute net input and apply step function for a single sample."""
        net = np.dot(self.weights, x) + self.bias
        return self._activate(net)

    def predict(self, X):
        """Predict class labels for a batch of samples."""
        return np.array([self.predict_one(x) for x in X])

    def train(self, X_train, y_train, epochs):
        """
        One pass = one epoch over all training samples.
        Each sample is presented in order; weights update immediately
        on any misclassification (online learning).
        """
        for epoch in range(epochs):
            n_errors = 0
            for x, target in zip(X_train, y_train):
                pred  = self.predict_one(x)
                error = target - pred          # 0 if correct, ±1 if wrong
                if error != 0:
                    self.weights += self.lr * error * x
                    self.bias    += self.lr * error
                    n_errors     += 1

            # Print progress every 10 epochs
            if (epoch + 1) % 10 == 0 or epoch == 0:
                preds    = self.predict(X_train)
                train_acc = np.mean(preds == y_train) * 100
                print(f"  Epoch {epoch+1:3d}/{epochs}  "
                      f"train_errors={n_errors}  train_acc={train_acc:.2f}%")


# 5. Evaluate 

def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred) * 100


# 6. Main pipeline

def main():
    TRAIN_PATH   = 'MushroomData_8000.txt'
    UNKNOWN_PATH = 'MushroomData_Unknwon_100.txt'   
    OUTPUT_PATH  = 'predictionResultPER.txt'

    # --- Load ---
    print("Loading data...")
    train_samples   = load_training(TRAIN_PATH)
    unknown_samples = load_unknown(UNKNOWN_PATH)
    print(f"  Training samples : {len(train_samples)}")
    print(f"  Unknown samples  : {len(unknown_samples)}")

    # --- Encode ---
    print("Building vocabulary and encoding features...")
    vocab       = build_vocab(train_samples)
    X_all, y_all = encode_dataset(train_samples, vocab, has_label=True)
    X_unknown, _ = encode_dataset(unknown_samples, vocab, has_label=False)
    n_features = X_all.shape[1]
    print(f"  Feature vector size (after one-hot): {n_features}")

    # --- Split ---
    X_train, y_train, X_val, y_val = train_val_split(
        X_all, y_all, TRAIN_RATIO, seed=RANDOM_SEED
    )
    print(f"  Train size: {len(y_train)}  |  Validation size: {len(y_val)}")

    # --- Train ---
    print(f"\nTraining Perceptron  (lr={LEARNING_RATE}, epochs={EPOCHS})...")
    model = Perceptron(n_features=n_features, learning_rate=LEARNING_RATE)
    model.train(X_train, y_train, epochs=EPOCHS)

    # --- Validate ---
    val_preds = model.predict(X_val)
    val_acc   = accuracy(y_val, val_preds)
    print(f"\nValidation accuracy: {val_acc:.2f}%")

    # --- Predict unknowns ---
    print("Predicting unknown mushrooms...")
    unknown_preds = model.predict(X_unknown)
    labels = ['e' if p == 1 else 'p' for p in unknown_preds]

    # --- Write output ---
    with open(OUTPUT_PATH, 'w') as f:
        f.write('\n'.join(labels) + '\n')
    print(f"Predictions written to '{OUTPUT_PATH}'")

    # Summary stats
    n_edible   = labels.count('e')
    n_poisonous = labels.count('p')
    print(f"  Edible: {n_edible}  |  Poisonous: {n_poisonous}")


if __name__ == '__main__':
    main()
