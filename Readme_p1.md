# CPSC 371 – Assignment 2: Mushroom Classification  
**Team:** The Unsupervised Guys  

Sukirat Singh Dhillon , 230155722  

Karsten Ngai Nakamura , 230165205

Akshay Arulkrishnan , 230158634

Amaan Hingora ,  

**Due:** March 23, 2026  
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

## Markdown Source

```markdown
## Overview  

The goal of this assignment was to build a **K-Nearest Neighbors (KNN)** classifier from scratch to predict whether a mushroom is edible (`e`) or poisonous (`p`), based on 22 categorical attributes.  

We were given:  
- `MushroomData_8000.txt` — 8000 labeled mushrooms for training  
- `MushroomData_Unknwon_100.txt` — 100 unlabeled mushrooms to classify  

The output file `predictionResultKNN.txt` contains one prediction per line (`e` or `p`) for all 100 unknown mushrooms.  

## How to Run  

Make sure Python 3 and matplotlib are installed.  

pip install matplotlib  
python nearestneighbor.py  

## Results  

| Metric              | Value   |
|---------------------|---------|
| Best K              | 1       |
| Validation accuracy | **100.00%** |
| Unknown – Edible    | 51      |
| Unknown – Poisonous | 49      |

## Efficiency  

| Metric                          | Value        |
|---------------------------------|-------------|
| K selection + validation time   | ~63 seconds |
| Unknown prediction time         | ~0.78 sec   |
| Average per prediction          | ~0.007 sec  |

## Visualization  

We generated a graph showing validation accuracy vs K.  

Saved as:  
knn_accuracy_plot.png  

## File Structure  

nearestneighbor.py  
predictionResultKNN.txt  
knn_accuracy_plot.png  
MushroomData_8000.txt  
MushroomData_Unknwon_100.txt  
Readme.md  

## Final Thoughts  

KNN performed extremely well and achieved perfect validation accuracy on this dataset.  
