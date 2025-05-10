import pandas as pd
import random

# Load the dataset
file_path = "DATASET_UPDATED.csv"  # Replace with the actual path to your file
df = pd.read_csv(file_path)

X = df.drop(columns=["UserName", "Review", "Label"])
y = df["Label"]

# Split the dataset into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

df_train = pd.concat([X_train, y_train], axis=1)
df_test = pd.concat([X_test, y_test], axis=1)

# Save the updated DataFrame to a new CSV file
df_train.to_csv("DATASET_TRAIN.csv", index=False)
df_test.to_csv("DATASET_TEST.csv", index=False)
