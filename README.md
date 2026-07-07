# DecodeLabs Data Science Internship — Project 2: Fraud Detection Pipeline

## Overview
A fraud detection pipeline built on 284,807 real-world credit card transactions.
The core challenge is extreme class imbalance — only 0.17% of transactions are 
fraudulent (577:1 ratio) — which breaks standard ML assumptions and makes 
accuracy a completely misleading metric.

## Dataset
- **Source:** Credit Card Fraud Detection — Kaggle (ULB Machine Learning Group)
- **Size:** 284,807 transactions | 31 columns
- **Fraud Rate:** 0.17% — 492 fraud cases out of 284,807 transactions

## What This Project Demonstrates
- Handling extreme class imbalance using SMOTE
- Building a leak-free pipeline with imbalanced-learn
- Training and tuning Logistic Regression and Random Forest classifiers
- Evaluating with Precision, Recall and ROC-AUC — not accuracy

## Results

| Model | Test ROC-AUC | Fraud Precision | Fraud Recall | False Alarms |
|---|---|---|---|---|
| Logistic Regression | 0.9715 | 6% | 92% | 1,448 |
| Random Forest ✅ | 0.9829 | 55% | 88% | 69 |

Random Forest was selected as the production model — achieving a 95% reduction 
in false alarms while maintaining strong fraud detection recall.

## Tech Stack
Python | Pandas | Scikit-learn | Imbalanced-learn | Matplotlib

## How to Run
```bash
git clone https://github.com/Rahmafynn/DecodeLabs-Internship.git
pip install pandas numpy scikit-learn imbalanced-learn matplotlib seaborn
```
Open `fraud_detection.ipynb` in Jupyter Notebook and run all cells.

## Author
Rahma — Data Science Intern, DecodeLabs Batch 2026
