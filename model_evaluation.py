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
        if "xg" in model_path.split("/")[-1].split(".")[0]:
            model = XGBoostScratch()
            model.trees = pickle.load(f)
        elif "random" in model_path.split("/")[-1].split(".")[0]:
            model = RandomForestScratch()
            model.trees = pickle.load(f)
        else:
            raise ValueError("Unrecognized model type in path.")
        return model

def load_numpy_model(model_path):
    if "logistic" in model_path.split("/")[-1].split(".")[0]:
        model = LogisticRegressionScratch()
        with open(model_path, "rb") as f:
            model.theta = np.load(f)
        return model
    else:
        raise ValueError("Unsupported .npy model type")

def evaluate_model(model, X_test, y_test):
    results = model.evaluate_test(X_test, y_test, verbose=False)
    if isinstance(results, dict):
        if "test_accuracy" in results:
            return results["test_accuracy"], results["precision"], results["recall"], results["f1_score"]
        elif "ensemble" in results and isinstance(results["ensemble"], dict):
            if "accuracy" in results["ensemble"]:
                return results["ensemble"]["accuracy"], results["ensemble"]["precision"], results["ensemble"]["recall"], results["ensemble"]["f1_score"]
            else:
                raise ValueError("Ensemble model does not contain accuracy")
        else:
            raise ValueError("Invalid results format returned by model")

def main():
    # Step 1: Load Test Dataset
    test_df = pd.read_csv("DATASET_TEST.csv")
    X_test = test_df.drop(columns=["Label"]).values
    y_test = test_df["Label"].astype(int).values  #Ensures indexing works in _cross_entropy

    # Step 2: Load Models
    log_model = load_numpy_model("Trained_Models_From_Scratch/logistic_regression.npy")
    rf_model = load_pickle_model("Trained_Models_From_Scratch/random_forest.pkl")
    xgb_model = load_pickle_model("Trained_Models_From_Scratch/xgboost.pkl")

    # Step 3: Evaluate Individual Models
    print("Evaluating Individual Models")
    print("-" * 25)
    
    log_metrics = evaluate_model(log_model, X_test, y_test)
    print(f"Logistic Regression Accuracy: {log_metrics[0]:.4f}")
    print(f"Logistic Regression Precision: {log_metrics[1]:.4f}")
    print(f"Logistic Regression Recall: {log_metrics[2]:.4f}")
    print(f"Logistic Regression F1 Score: {log_metrics[3]:.4f}")

    rf_metrics = evaluate_model(rf_model, X_test, y_test)
    print(f"\nRandom Forest Accuracy: {rf_metrics[0]:.4f}")
    print(f"Random Forest Precision: {rf_metrics[1]:.4f}")
    print(f"Random Forest Recall: {rf_metrics[2]:.4f}")
    print(f"Random Forest F1 Score: {rf_metrics[3]:.4f}")

    xgb_metrics = evaluate_model(xgb_model, X_test, y_test)
    print(f"\nXGBoost Accuracy: {xgb_metrics[0]:.4f}")
    print(f"XGBoost Precision: {xgb_metrics[1]:.4f}")
    print(f"XGBoost Recall: {xgb_metrics[2]:.4f}")
    print(f"XGBoost F1 Score: {xgb_metrics[3]:.4f}")

    # Step 4: Ensemble Voting
    print("\n\nEvaluating Ensemble Model")
    print("-" * 25)
    ensemble_model = EnsembleModel(models=[log_model, rf_model, xgb_model])
    ensemble_metrics = evaluate_model(ensemble_model, X_test, y_test)
    
    print(f"Ensemble Model Accuracy: {ensemble_metrics[0]:.4f}")
    print(f"Ensemble Model Precision: {ensemble_metrics[1]:.4f}")
    print(f"Ensemble Model Recall: {ensemble_metrics[2]:.4f}")
    print(f"Ensemble Model F1 Score: {ensemble_metrics[3]:.4f}")

  

    # Step 5: Visualize the results
    import matplotlib.pyplot as plt
    import seaborn as sns

    model_names = ["Logistic Regression", "Random Forest", "XGBoost", "Ensemble"]
    accuracies = [log_metrics[0], rf_metrics[0], xgb_metrics[0], ensemble_metrics[0]]

    plt.figure(figsize=(10, 6))
    sns.barplot(x=model_names, y=accuracies, )
    plt.title("Model Accuracies")
    plt.ylabel("Accuracy")
    plt.xlabel("Models")
    plt.ylim(0, 1)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("model_accuracies.png")
    plt.show()

    


if __name__ == "__main__":
    main()
