import numpy as np
import pandas as pd
import pickle
import os

from Logistic_Regression_Scratch import LogisticRegressionScratch
from random_forest_classifier_scratch import RandomForestScratch
from xgboost_scratch import XGBoostScratch

def save_logistic_regression(model, path):
    np.save(path, model.theta)

def save_random_forest(model, path):
    with open(path, 'wb') as f:
        pickle.dump(model.trees, f)

def save_xgboost(model, path):
    with open(path, 'wb') as f:
        pickle.dump(model.trees, f)

def main():
    # Step 1: Load Training Dataset
    train_df = pd.read_csv("DATASET_TRAIN.csv")
    X_train = train_df.drop(columns=["Label"]).values
    y_train = train_df["Label"].values

    # Step 2: Make sure saved_models/ directory exists
    if not os.path.exists("Trained_Models_From_Scratch"):
        os.makedirs("Trained_Models_From_Scratch\n")

    # Step 3: Train Logistic Regression (50 iterations)
    print("Training Logistic Regression")
    print("-" * 25)
    log_model = LogisticRegressionScratch(learning_rate=0.1, num_iterations=50)
    log_model.fit(X_train, y_train)
    save_logistic_regression(log_model, "Trained_Models_From_Scratch/logistic_regression.npy")
    print("Logistic Regression Done\n")

    # Step 4: Train Random Forest (50 trees for faster demo)
    print("Training Random Forest")
    print("-" * 25)
    rf_model = RandomForestScratch(n_trees=50, max_depth=5)
    rf_model.fit(X_train, y_train)
    save_random_forest(rf_model, "Trained_Models_From_Scratch/random_forest.pkl")
    print("Random Forest Done\n")

    # Step 5: Train XGBoost (50 boosting rounds)
    print("Training XGBoost")
    print("-" * 25)
    xgb_model = XGBoostScratch(n_estimators=50, learning_rate=0.1)
    xgb_model.fit(X_train, y_train)
    save_xgboost(xgb_model, "Trained_Models_From_Scratch/xgboost.pkl")
    print("XGBoost model Done")

    import matplotlib.pyplot as plt
    # Step 6:  Same for training loss
    plt.figure(figsize=(10, 6))
    plt.scatter(range(len(log_model.train_loss_history)), log_model.train_loss_history, label="Logistic Regression", color="blue")
    plt.scatter(range(len(rf_model.train_loss_history)), rf_model.train_loss_history, label="Random Forest", color="green")
    plt.scatter(range(len(xgb_model.train_loss_history)), xgb_model.train_loss_history, label="XGBoost", color="red")
    plt.title("Training Loss")
    plt.ylabel("Training Loss")
    plt.xlabel("Epochs")
    plt.legend()
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig("training_loss.png")
    plt.show()

    # Step 6:  Same for validation loss
    plt.figure(figsize=(10, 6))
    plt.scatter(range(len(log_model.val_loss_history)), log_model.val_loss_history, label="Logistic Regression", color="blue")
    plt.scatter(range(len(rf_model.val_loss_history)), rf_model.val_loss_history, label="Random Forest", color="green")
    plt.scatter(range(len(xgb_model.val_loss_history)), xgb_model.val_loss_history, label="XGBoost", color="red")
    plt.title("Validation Loss")
    plt.ylabel("Validation Loss")
    plt.xlabel("Epochs")
    plt.legend()
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig("validation_loss.png")
    plt.show()


    








if __name__ == "__main__":
    main()
