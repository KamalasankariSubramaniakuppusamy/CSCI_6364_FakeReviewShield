import numpy as np
import pandas as pd
import pickle
import os

from xgboost_scratch import XGBoostScratch
from random_forest_classifier_scratch import RandomForestScratch
from Logistic_Regression_Scratch import LogisticRegressionScratch
from ensemble_voting import EnsembleModel

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

def evaluate_model(model, X_test, y_test):
    results = model.evaluate_test(X_test, y_test, verbose=False)
    if isinstance(results, dict):
        if model.__class__.__name__ == "EnsembleModel":
            accuracy = results["ensemble"]["accuracy"]
        else:
            accuracy = results["test_accuracy"]
    else:
        raise ValueError("Invalid results format")
    return accuracy

def main():
    # Step 1: Load Test Dataset
    test_df = pd.read_csv("test.csv")
    X_test = test_df.drop(columns=["Label"]).values
    y_test = test_df["Label"].values

    # Step 2: Load Models
    log_model = load_numpy_model("Trained_Models_From_Scratch/logistic_regression.npy")
    rf_model = load_pickle_model("Trained_Models_From_Scratch/random_forest.pkl")
    xgb_model = load_pickle_model("Trained_Models_From_Scratch/xgboost.pkl")

    # Step 3: Evaluate Individual Models
    print("Evaluating Individual Models")
    print("-" * 25)
    
    log_accuracy = evaluate_model(log_model, X_test, y_test)
    print(f"Logistic Regression Accuracy: {log_accuracy:.4f}")

    rf_accuracy = evaluate_model(rf_model, X_test, y_test)
    print(f"Random Forest Accuracy: {rf_accuracy:.4f}")

    xgb_accuracy = evaluate_model(xgb_model, X_test, y_test)
    print(f"XGBoost Accuracy: {xgb_accuracy:.4f}")

    # Step 4: Ensemble Voting
    print("\nEvaluating Ensemble Model")
    print("-" * 25)
    ensemble_model = EnsembleModel(models=[log_model, rf_model, xgb_model])
    ensemble_accuracy = evaluate_model(ensemble_model, X_test, y_test)
    
    print(f"Ensemble Model Accuracy: {ensemble_accuracy:.4f}")

    # Visualize the results
    import matplotlib.pyplot as plt
    import seaborn as sns

    model_names = ["Logistic Regression", "Random Forest", "XGBoost", "Ensemble"]
    accuracies = [log_accuracy, rf_accuracy, xgb_accuracy, ensemble_accuracy]

    plt.figure(figsize=(10, 6))
    sns.barplot(x=model_names, y=accuracies, palette="viridis")
    plt.title("Model Accuracies")
    plt.ylabel("Accuracy")
    plt.xlabel("Models")
    plt.ylim(0, 1)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()