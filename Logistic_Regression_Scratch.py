import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score

class LogisticRegressionScratch:
    def __init__(self, learning_rate=0.1, num_iterations=200):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.theta = None
        self.train_loss_history = []
        self.val_loss_history = []

    def sigmoid(self, z):
        z = np.clip(z, -500, 500)  # Avoid overflow
        return 1 / (1 + np.exp(-z))

    def compute_loss(self, h, y):
        epsilon = 1e-15  # Avoid log(0)
        h = np.clip(h, epsilon, 1 - epsilon)
        return (-1 / y.size) * (np.dot(y, np.log(h)) + np.dot((1 - y), np.log(1 - h)))

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
        X_train = np.c_[np.ones((X_train.shape[0], 1)), X_train]
        if X_val is not None:
            X_val = np.c_[np.ones((X_val.shape[0], 1)), X_val]

        self.theta = np.zeros(X_train.shape[1])

        for i in range(self.num_iterations):
            # Forward pass for training
            z_train = np.dot(X_train, self.theta)
            h_train = self.sigmoid(z_train)
            gradient = np.dot(X_train.T, (h_train - y_train)) / y_train.size
            self.theta -= self.learning_rate * gradient

            # Train metrics
            train_loss = self.compute_loss(h_train, y_train)
            train_pred = (h_train >= 0.5).astype(int)
            train_acc = np.mean(train_pred == y_train)
            self.train_loss_history.append(train_loss)

            # Validation metrics
            if X_val is not None and y_val is not None:
                h_val = self.sigmoid(np.dot(X_val, self.theta))
                val_loss = self.compute_loss(h_val, y_val)
                val_pred = (h_val >= 0.5).astype(int)
                val_acc = np.mean(val_pred == y_val)
                self.val_loss_history.append(val_loss)

                print(f"Iter {i+1}/{self.num_iterations}: "
                      f"Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f} | "
                      f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            else:
                print(f"Iter {i+1}/{self.num_iterations}: "
                      f"Train Acc: {train_acc:.4f} | Train Loss: {train_loss:.4f}")

    def predict_prob(self, X):
        X = np.c_[np.ones((X.shape[0], 1)), X]
        return self.sigmoid(np.dot(X, self.theta))

    def predict(self, X):
        return (self.predict_prob(X) >= 0.5).astype(int)

    def predict_from_user_input(self, user_input_vector):
        return self.predict(user_input_vector.reshape(1, -1))[0]

    def evaluate_test(self, X_test, y_test, verbose=True):
        y_pred = self.predict(X_test)
        accuracy = np.mean(y_pred == y_test)
        error = np.mean(y_pred != y_test)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        if verbose:
            print(f"Test Accuracy: {accuracy:.4f}")
            print(f"Test Error: {error:.4f}")
            print(f"Test Precision: {precision:.4f}")
            print(f"Test Recall: {recall:.4f}")
            print(f"Test F1 Score: {f1:.4f}")
        return {
            "test_accuracy": accuracy,
            "test_error": error,
            "predictions": y_pred,
            "precision": precision,
            "recall": recall,
            "f1_score": f1
        }
