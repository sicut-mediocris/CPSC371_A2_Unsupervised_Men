# CPSC 371 – Assignment 2: Mushroom Classification
**Team:** The Unsupervised Guys

Sukirat Singh Dhillon , 230155722

Karsten Ngai Nakamura , 230165205

Akshay Arulkrishnan , 230158634

Amaan Hingora , 230156282

**Due:** March 23, 2026

---

Disclaimer : AI assistance was used to edit the markdown file and format code

---

## Part 1 – K-Nearest Neighbors (KNN)

---

## Overview

The goal of this assignment was to build a **K-Nearest Neighbors (KNN)** classifier from scratch to predict whether a mushroom is edible (`e`) or poisonous (`p`), based on 22 categorical attributes.

We were given:
- `MushroomData_8000.txt` — 8000 labeled mushrooms for training
- `MushroomData_Unknwon_100.txt` — 100 unlabeled mushrooms to classify

The output file `predictionResultKNN.txt` contains one prediction per line (`e` or `p`) for all 100 unknown mushrooms.

---

## How to Run

Make sure Python 3 and matplotlib are installed.

pip install matplotlib  
python nearestneighbor.py

---

## Thought Process & Approach

### Step 1 – Understanding the Data

Each row contains 23 values:
- First value → label (`e` or `p`)
- Remaining 22 → categorical features

All features are categorical, so we cannot use numerical distance like Euclidean distance.

---

### Step 2 – Distance Function

We used a **mismatch distance (Hamming distance)**:

distance = number of positions where values differ

This works well for categorical data.

---

### Step 3 – KNN Algorithm

For each unknown mushroom:
1. Compute distance to all training samples
2. Sort distances
3. Take K nearest neighbors
4. Count edible vs poisonous
5. Assign majority label

If tie:
→ Use weighted voting based on distance

---

### Step 4 – Choosing Best K

We tested:

[1, 3, 5, 7, 9, 11]

Process:
- Split data → 80% train / 20% validation
- Evaluate accuracy for each K
- Pick best K

---

## Results

| Metric              | Value   |
|---------------------|---------|
| Best K              | 1       |
| Validation accuracy | 100.00% |
| Unknown – Edible    | 51      |
| Unknown – Poisonous | 49      |

---

## Efficiency

| Metric                          | Value        |
|---------------------------------|-------------|
| K selection + validation time   | ~63 seconds |
| Unknown prediction time         | ~0.78 sec   |
| Total runtime                   | ~64 sec     |
| Avg per prediction              | ~0.007 sec  |

---

## Visualization

A graph was generated showing validation accuracy vs K.

Saved as:
knn_accuracy_plot.png

---

## File Structure

nearestneighbor.py  
predictionResultKNN.txt  
knn_accuracy_plot.png  
MushroomData_8000.txt  
MushroomData_Unknwon_100.txt  
Readme_p1.md  

---

## Notes

- No external ML libraries used
- Only Python + matplotlib
- Distance metric: mismatch (categorical-safe)
- KNN stores full dataset (no training phase)

---

## Final Thoughts

KNN performed extremely well and achieved perfect validation accuracy on this dataset.

The dataset appears to be highly separable, which explains why even K=1 performs optimally.

The main drawback is efficiency:
- KNN is computationally expensive
- Every prediction requires scanning all training data

---
