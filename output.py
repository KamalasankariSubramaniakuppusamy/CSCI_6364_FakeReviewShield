import numpy as np
import pandas as pd
import pickle

from Logistic_Regression_Scratch import LogisticRegressionScratch
from random_forest_classifier_scratch import RandomForestScratch
from xgboost_scratch import XGBoostScratch
from ensemble_voting import EnsembleModel
from database_manager import UserManager
from review_analysis import ReviewAnalyzer

def load_pickle_model(model_path):
    with open(model_path, "rb") as f:
        if "xg" in model_path.split("/")[1].split(".")[0]:
            model = XGBoostScratch()
            model.trees = pickle.load(f)
        elif "random" in model_path.split("/")[1].split(".")[0]:
            model = RandomForestScratch()
            model.trees = pickle.load(f)
        return model

def load_numpy_model(model_path):
    if "logistic" in model_path.split("/")[1].split(".")[0]:
        model = LogisticRegressionScratch()
        with open(model_path, "rb") as f:
            model.theta = np.load(f)
    else:
        raise ValueError("Unsupported model type")
    return model

username = input("Enter your Username: ").strip()
review_text = input("Enter your Review Text: ").strip()
rating = float(input("Enter your Rating (1-5): ").strip())

user_manager = UserManager()
user_features = user_manager.lookup_or_create_user(username)

review_analyzer = ReviewAnalyzer(user_features)
review_features = review_analyzer.analyze_review(review_text, rating)

feature_vector1 = np.array(review_features)
feature_vector1 = feature_vector1.reshape(1, -1)

log_model = load_numpy_model("Trained_Models_From_Scratch/logistic_regression.npy")
rf_model = load_pickle_model("Trained_Models_From_Scratch/random_forest.pkl")
xgb_model = load_pickle_model("Trained_Models_From_Scratch/xgboost.pkl")

model = EnsembleModel(models=[log_model, rf_model, xgb_model])

# Predicting the review from input (most likely to be fake since the purchase is not verified)
prediction = model.predict(feature_vector1)
if prediction == 1:
    print("The review is likely to be fake.")
else:
    print("The review is likely to be real.")
