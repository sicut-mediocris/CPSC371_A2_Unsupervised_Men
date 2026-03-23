# CPSC 371 – Assignment 2: Mushroom Classification
**Team:** Unsupervised Men
**Due:** March 23, 2026

---

## Overview

The goal of this assignment was to build a **Perceptron** classifier from scratch to predict whether a mushroom is edible (`e`) or poisonous (`p`), based on 22 categorical attributes.

We were given:
- `MushroomData_8000.txt` — 8000 labeled mushrooms for training
- `MushroomData_Unknwon_100.txt` — 100 unlabeled mushrooms to classify

The output file `predictionResultPER.txt` contains one prediction per line (`e` or `p`) for all 100 unknown mushrooms.

---

## How to Run

Make sure Python 3 and numpy are installed.

```bash
pip install numpy
python perceptron.py
```

This will:
1. Load and preprocess the training data
2. Train the perceptron
3. Evaluate accuracy on the validation split
4. Write predictions to `predictionResultPER.txt`

---

## Thought Process & Approach

### Step 1 – Understanding the Data

Each row in the training file has 23 comma-separated values. The first value is the class label (`e` or `p`), and the remaining 22 are categorical attributes like cap shape, odor, gill color, etc. Some values are missing and represented as `?`.

Since everything is categorical (no numeric values), we couldn't just feed raw letters into the perceptron — we had to convert them to numbers first.

### Step 2 – Preprocessing (One-Hot Encoding)

We went with **one-hot encoding** for each attribute. For example, if cap-shape can be `b, c, x, f, k, s`, we turn each value into a 6-element binary vector where exactly one position is `1`. This way, the perceptron sees meaningful numeric inputs.

For missing values (`?`), instead of dropping those rows or guessing a value, we treated `?` as its own valid category and gave it its own column. This felt more honest than imputing values we don't know.

After encoding all 22 attributes, each mushroom became a **117-dimensional binary vector**.

We fit the vocabulary (which values exist per column) on the training data only, then applied the same encoding to the unknown test mushrooms. This avoids any data leakage.

### Step 3 – Train/Validation Split

We split the 8000 training samples **80% train / 20% validation** (6400 train, 1600 validation), shuffled randomly with a fixed seed for reproducibility.

We chose 80/20 because it gives enough data to train while still having a meaningful validation set to measure real performance.

### Step 4 – Building the Perceptron

The perceptron is implemented from scratch using only numpy. Here's the core idea:

- **Weights** initialized to small random values in `[-0.05, 0.05]` (avoids all-zero symmetry)
- **Bias** initialized to `0`
- **Activation:** step function — output `1` (edible) if net input ≥ 0, else `0` (poisonous)
- **Learning rule (online):** on each misclassification:
  ```
  weights += learning_rate × (target - prediction) × x
  bias    += learning_rate × (target - prediction)
  ```
  Correct predictions → no update

This is standard online (sample-by-sample) perceptron learning, not batch gradient descent.

### Step 5 – Choosing Hyperparameters

We experimented a bit and settled on:

| Parameter     | Value |
|---------------|-------|
| Learning rate | 0.01  |
| Epochs        | 100   |
| Train split   | 80%   |
| Random seed   | 42    |

We started with `lr=0.1` but found `0.01` gave cleaner convergence. The model converged to 0 training errors by **epoch 10**, so 100 epochs is more than enough.

---

## Results

| Metric              | Value   |
|---------------------|---------|
| Training accuracy   | 100.00% |
| Validation accuracy | **100.00%** |
| Unknown – Edible    | 51      |
| Unknown – Poisonous | 49      |

The dataset turns out to be **linearly separable** after one-hot encoding, which is why the perceptron achieves perfect accuracy. This makes sense — mushroom edibility has clear distinguishing features (odor alone is a very strong predictor).

---

## File Structure

```
perceptron.py                  ← main implementation
predictionResultPER.txt        ← output predictions for 100 unknown mushrooms
MushroomData_8000.txt          ← training data (8000 labeled mushrooms)
MushroomData_Unknwon_100.txt   ← test data (100 unlabeled mushrooms)
Readme.md                      ← this file
```

---

## Notes

- No external ML libraries were used (no scikit-learn, no keras, etc.)
- Only `numpy` and Python built-ins were used
- The perceptron is a **binary classifier**: edible = 1, poisonous = 0
