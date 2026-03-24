"""
Mushroom Classification using K-Nearest Neighbors
CPSC 371 - Assignment 2

Sukirat Singh Dhillon,230155722
Karsten Ngai Nakamura
Amaan Hingora, 230156282
Akshay ArulKrishnan

All attributes are categorical. For KNN, distance will be based on how many
attribute values differ between two mushrooms. Missing values ('?') are treated
as normal categorical values.
"""

import numpy as np
import random

# Hyperparameters
K             = 5
TRAIN_RATIO   = 0.80   # 80% train, 20% validation
RANDOM_SEED   = 42

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# 1. Load raw data

def load_training(path):
    """Read training file. Returns (X, y) where y is 'e' or 'p'."""
    X, y = [], []
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(',')
            y.append(parts[0])      # 'e' or 'p'
            X.append(parts[1:])     # 22 categorical attributes
    return np.array(X), np.array(y)


def load_unknown(path):
    """Read unknown file. Returns X only (22 attributes, no label)."""
    X = []
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            X.append(line.split(','))
    return np.array(X)


# 2. Train / validation split

def train_val_split(X, y, train_ratio, seed=42):
    """Shuffle and split into training and validation sets."""
    n = len(y)
    indices = list(range(n))
    random.Random(seed).shuffle(indices)
    split = int(n * train_ratio)
    train_idx = indices[:split]
    val_idx   = indices[split:]
    return X[train_idx], y[train_idx], X[val_idx], y[val_idx]


# 3. Evaluate

def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred) * 100


# 4. Main pipeline

def main():
    TRAIN_PATH   = 'MushroomData_8000.txt'
    UNKNOWN_PATH = 'MushroomData_Unknwon_100.txt'

    # --- Load ---
    print("Loading data...")
    X_all, y_all = load_training(TRAIN_PATH)
    X_unknown    = load_unknown(UNKNOWN_PATH)

    print(f"  Training samples : {len(X_all)}")
    print(f"  Unknown samples  : {len(X_unknown)}")
    print(f"  Attributes per sample: {X_all.shape[1]}")

    # --- Split ---
    X_train, y_train, X_val, y_val = train_val_split(
        X_all, y_all, TRAIN_RATIO, seed=RANDOM_SEED
    )
    print(f"  Train size: {len(y_train)}  |  Validation size: {len(y_val)}")

    # Quick label summary
    n_edible = np.sum(y_all == 'e')
    n_poisonous = np.sum(y_all == 'p')
    print(f"  Edible: {n_edible}  |  Poisonous: {n_poisonous}")


if __name__ == '__main__':
    main()