# Fake News Detector

![Python](https://img.shields.io/badge/Python-3.x-blue) ![Accuracy](https://img.shields.io/badge/Accuracy-98.8%25-brightgreen) ![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange)

A binary text classifier that labels news articles as real or fake using TF-IDF features and logistic regression.

---

## The Problem

Misinformation spreads fast online. I wanted to build a model that could flag suspicious articles automatically — no manual fact-checking required.

---

## Results

| Split | Accuracy |
|-------|----------|
| Train | 99.17% |
| Test | 98.84% |
| Holdout | 99.50% |

---

## Why Logistic Regression?

I tested both Logistic Regression and Naive Bayes. Naive Bayes was faster to train but less accurate. Logistic Regression gave the better accuracy-to-speed tradeoff and is more interpretable — you can inspect which words pushed the prediction in either direction.

---

## What Broke — and What I Learned

When I tested the model on newer articles outside the training data, accuracy dropped noticeably. I diagnosed this as **distribution shift** — the model learned patterns specific to the dataset's time period rather than general fake-news signals.

This is a known limitation of static datasets in NLP and would require periodic retraining or a more robust feature set to solve properly.

---

## Stack

- Python
- scikit-learn
- pandas
- joblib
- TF-IDF (TfidfVectorizer)

---

## Dataset

Trained on the [Fake and Real News Dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset) from Kaggle.