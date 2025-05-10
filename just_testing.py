import numpy as np
import pandas as pd
import pickle

from xgboost_scratch import XGBoostScratch
from random_forest_classifier_scratch import RandomForestScratch
from Logistic_Regression_Scratch import LogisticRegressionScratch

test_df = pd.read_csv("test.csv")
X_test = test_df.drop(columns=["Label"]).values
y_test = test_df["Label"].values

# USING LOGISTIC REGRESSION
print("Testing Logistic Regression")
print("-" * 25)
model = LogisticRegressionScratch(learning_rate=0.1, num_iterations=50)

with open("./Trained_Models_From_Scratch/logistic_regression.npy", "rb") as f:
    model.theta = np.load(f)
model.evaluate_test(X_test, y_test, verbose=True)

# USING RF
print("\nTesting Random Forest")
print("-" * 25)
model = RandomForestScratch(n_trees=10, max_depth=5)

with open("./Trained_Models_From_Scratch/random_forest.pkl", "rb") as f:
    model.trees = pickle.load(f)
model.evaluate_test(X_test, y_test, verbose=True)

# USING XGBOOST
print("\nTesting XGBoost")
print("-" * 25)

model = XGBoostScratch(n_estimators=10, learning_rate=0.1)

with open("./Trained_Models_From_Scratch/xgboost.pkl", "rb") as f:
    model.trees = pickle.load(f)
model.evaluate_test(X_test, y_test, verbose=True)