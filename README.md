# FakeReviewShield

**FakeReviewShield** is a machine learning-based system designed to detect fake online reviews. It classifies reviews as either **Original (0)** or **Fake (1)** using models implemented from scratch and an ensemble voting mechanism for enhanced reliability.

## Overview

Fake reviews are a growing threat to consumer trust on e-commerce platforms. This project addresses that problem by deploying a predictive model pipeline trained on a curated review dataset.

## Dataset

We used the Kaggle **Amazon Product Review Spam and Non-Spam** dataset, containing labeled reviews from various categories. After preprocessing, a final set of 150,000 balanced reviews was used for model training and evaluation.

## Features

The feature set combines textual and behavioral data:

* Review length
* Rating (1–5)
* Incentivized keywords
* Reviewer history
* Upvotes
* Presence of photos/videos

TF-IDF vectorization was used for text, and metadata features were integer-encoded.

## Models

Implemented from scratch:

* Logistic Regression
* Random Forest
* XGBoost

An ensemble voting classifier aggregates predictions to improve overall accuracy and reduce bias.

## Performance

| Model               | Accuracy | Precision | Recall | F1-score | Validation Loss |
| ------------------- | -------- | --------- | ------ | -------- | --------------- |
| Logistic Regression | 93.26%   | 0.9015    | 0.9785 | 0.9384   | 0.4887          |
| Random Forest       | 97.51%   | 0.9864    | 0.9672 | 0.9767   | 0.2157          |
| XGBoost             | 97.60%   | 0.9880    | 0.9672 | 0.9775   | 0.2462          |
| Ensemble Voting     | 97.70%   | 0.9864    | 0.9672 | 0.9767   | -               |

## How to Use

1. **Fork the Repository**
   Go to: [https://github.com/KamalasankariSubramaniakuppusamy/CSCI\_6364\_FakeReviewShield](https://github.com/KamalasankariSubramaniakuppusamy/CSCI_6364_FakeReviewShield) and click "Fork".

2. **Train the Models**
   Run the training script to train all models from scratch:

```bash
python model_training.py
```

3. **Evaluate the Models**
   Evaluate model performance and compute metrics:

```bash
python model_evaluation.py
```

4. **Run Deployment Interface**
   Launch the deployed system to input and classify reviews:

```bash
python output.py
```

Ensure all dependencies (such as numpy, pandas) are installed, and that dataset files are properly placed.

## Contributors

* Kamalasankari Subramaniakuppusamy – G43816454
* Manali Moger – G38979983
* Veditha Reddy Avuthu – G43696437
