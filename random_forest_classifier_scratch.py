import numpy as np
import random
from sklearn.model_selection import train_test_split
from collections import Counter
from math import log
from sklearn.metrics import precision_score, recall_score, f1_score

class DecisionTreeScratch:
    def __init__(self, max_depth=5, min_samples_split=2, feature_sample_ratio=0.7):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.feature_sample_ratio = feature_sample_ratio
        self.tree = None

    def fit(self, X, y):
        data = np.c_[X, y]
        self.tree = self._build_tree(data)

    def _build_tree(self, train):
        root = self._get_split(train)
        self._split_node(root, 1)
        return root

    def _split_node(self, node, depth):
        left, right = node['groups']
        del(node['groups'])

        if len(left) == 0 or len(right) == 0:
            node['left'] = node['right'] = self._to_terminal(np.vstack((left, right)))
            return

        if depth >= self.max_depth:
            node['left'], node['right'] = self._to_terminal(left), self._to_terminal(right)
            return

        if len(left) <= self.min_samples_split:
            node['left'] = self._to_terminal(left)
        else:
            node['left'] = self._get_split(left)
            self._split_node(node['left'], depth+1)

        if len(right) <= self.min_samples_split:
            node['right'] = self._to_terminal(right)
        else:
            node['right'] = self._get_split(right)
            self._split_node(node['right'], depth+1)

    def _gini(self, groups, classes):
        n_instances = sum([len(group) for group in groups])
        gini = 0.0
        for group in groups:
            if len(group) == 0:
                continue
            score = 0.0
            group_labels = group[:, -1]
            for class_val in classes:
                p = np.sum(group_labels == class_val) / len(group)
                score += p * p
            gini += (1.0 - score) * (len(group) / n_instances)
        return gini

    def _get_split(self, dataset):
        class_values = np.unique(dataset[:, -1])
        b_index, b_value, b_score, b_groups = None, None, float('inf'), None
        
        n_features = dataset.shape[1] - 1
        feature_indices = random.sample(
            range(n_features), 
            max(1, int(n_features * self.feature_sample_ratio)))
        
        for index in feature_indices:
            values = np.unique(dataset[:, index])
            for value in values:
                groups = self._split(index, value, dataset)
                gini = self._gini(groups, class_values)
                if gini < b_score:
                    b_index, b_value, b_score, b_groups = index, value, gini, groups
        return {'index': b_index, 'value': b_value, 'groups': b_groups}

    def _split(self, index, value, dataset):
        left = dataset[dataset[:, index] < value]
        right = dataset[dataset[:, index] >= value]
        return left, right

    def _to_terminal(self, group):
        outcomes = group[:, -1].astype(int)
        return Counter(outcomes).most_common(1)[0][0]

    def predict(self, X):
        return np.array([self._predict_sample(sample, self.tree) for sample in X])

    def _predict_sample(self, row, node):
        if row[node['index']] < node['value']:
            if isinstance(node['left'], dict):
                return self._predict_sample(row, node['left'])
            else:
                return node['left']
        else:
            if isinstance(node['right'], dict):
                return self._predict_sample(row, node['right'])
            else:
                return node['right']

class RandomForestScratch:
    def __init__(self, n_trees=100, max_depth=None, min_samples_split=2,
                 sample_size_ratio=0.8, feature_sample_ratio=0.3, 
                 test_size=0.2, random_state=None):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.sample_size_ratio = sample_size_ratio
        self.feature_sample_ratio = feature_sample_ratio
        self.test_size = test_size
        self.random_state = random_state
        self.trees = []
        self.classes_ = None
        self.train_acc_history = []
        self.val_acc_history = []
        self.train_loss_history = []
        self.val_loss_history = []

    def _subsample(self, X, y):
        n_samples = round(X.shape[0] * self.sample_size_ratio)
        indices = np.random.choice(X.shape[0], n_samples, replace=True)
        return X[indices], y[indices]

    def _cross_entropy(self, y_true, y_prob):
        epsilon = 1e-15  # to avoid log(0)
        y_prob = np.clip(y_prob, epsilon, 1 - epsilon)
        n_samples = y_true.shape[0]
        
        # Convert y_true to indices that match the columns of y_prob
        # First, ensure y_true contains valid class labels
        y_true_indices = np.array([np.where(self.classes_ == label)[0][0] for label in y_true])
        
        log_likelihood = -np.log(y_prob[np.arange(n_samples), y_true_indices])
        return np.sum(log_likelihood) / n_samples

    def fit(self, X, y, verbose=True):
        np.random.seed(self.random_state)
        random.seed(self.random_state)
        
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=self.test_size, 
            stratify=y, random_state=self.random_state
        )

        self.trees = []
        for i in range(self.n_trees):
            # Train new tree
            X_sample, y_sample = self._subsample(X_train, y_train)
            tree = DecisionTreeScratch(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                feature_sample_ratio=self.feature_sample_ratio
            )
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)

            # Calculate metrics
            train_prob = self.predict_prob(X_train)
            train_acc = np.mean(self._predict(X_train) == y_train)
            train_loss = self._cross_entropy(y_train, train_prob)
            
            val_prob = self.predict_prob(X_val)
            val_acc = np.mean(self._predict(X_val) == y_val)
            val_loss = self._cross_entropy(y_val, val_prob)
            
            # Store history
            self.train_acc_history.append(train_acc)
            self.val_acc_history.append(val_acc)
            self.train_loss_history.append(train_loss)
            self.val_loss_history.append(val_loss)

            # Print progress
            if verbose and ((i+1) % 1 == 0 or i == 0 or (i+1) == self.n_trees):
                print(f"Iter {i+1}/{self.n_trees}: "
                      f"Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f} | "
                      f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

    def _predict(self, X):
        preds = np.array([tree.predict(X) for tree in self.trees])
        return np.apply_along_axis(
            lambda x: Counter(x).most_common(1)[0][0], 
            axis=0, 
            arr=preds
        )

    def predict(self, X):
        return self._predict(X)

    def predict_prob(self, X):
        if not self.trees:
            return np.zeros((X.shape[0], len(self.classes_)))
        
        preds = np.array([tree.predict(X) for tree in self.trees])
        prob = np.zeros((X.shape[0], len(self.classes_)))
        for i, cls in enumerate(self.classes_):
            prob[:, i] = np.mean(preds == cls, axis=0)
        return prob

    def score(self, X, y):
        return np.mean(self.predict(X) == y)

    def get_metrics_history(self):
        return {
            'train_accuracy': self.train_acc_history,
            'val_accuracy': self.val_acc_history,
            'train_loss': self.train_loss_history,
            'val_loss': self.val_loss_history
        }
    
    def evaluate_test(self, X_test, y_test, verbose=True):
        if not self.trees:
            raise RuntimeError("Model not trained yet - call fit() first")
        
        self.classes_ = np.unique(y_test)
        
        # Calculate predictions and probabilities
        test_pred = self._predict(X_test)
        test_prob = self.predict_prob(X_test)
        
        # Calculate metrics
        test_acc = np.mean(test_pred == y_test)
        test_loss = self._cross_entropy(y_test, test_prob)
        test_precision = precision_score(y_test, test_pred)
        test_recall = recall_score(y_test, test_pred)
        test_f1 = f1_score(y_test, test_pred)

        if verbose:
            print("Test Set Metrics:")
            print(f"Test Loss: {test_loss:.4f}")
            print(f"Test Accuracy: {test_acc:.4f}")
            print(f"Test Precision: {test_precision:.4f}")
            print(f"Test Recall: {test_recall:.4f}")
            print(f"Test F1 Score: {test_f1:.4f}")
        
        return {
            'test_accuracy': test_acc,
            'loss': test_loss,
            'predictions': test_pred,
            'probabilities': test_prob,
            'precision': test_precision,
            'recall': test_recall,
            'f1_score': test_f1
        }
